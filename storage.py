# Постоянное хранилище результатов: SQLite через aiosqlite.
#
# Интерфейс функций остался близким к лабе №1
# (save_prediction, list_predictions), но теперь они асинхронные:
# блокирующий sqlite3 внутри async-endpoint заморозил бы event loop.

import logging
import math
import uuid
from contextlib import asynccontextmanager
from typing import List, Optional

import aiosqlite
from fastapi import HTTPException, status

import config
import model
from services import get_recommendation, get_risk_level

logger = logging.getLogger(__name__)


INSERT_SQL = """
    INSERT INTO predictions (
        request_id, farm_id, risk_score, risk_level,
        recommendation, model_version, source
    )
    VALUES (?, ?, ?, ?, ?, ?, ?);
"""

SELECT_COLUMNS = """
    request_id, farm_id, risk_score, risk_level,
    recommendation, model_version
"""


def storage_unavailable() -> HTTPException:
    """Единая ошибка 503 для любых сбоев БД."""

    return HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail="Prediction storage is unavailable"
    )


@asynccontextmanager
async def get_db():
    """
    Отдельное асинхронное соединение с SQLite
    на одну логическую операцию.
    """

    db = await aiosqlite.connect(
        str(config.DATABASE_PATH),
        timeout=config.DB_BUSY_TIMEOUT_MS / 1000
    )

    db.row_factory = aiosqlite.Row

    # busy_timeout задается для каждого нового соединения
    await db.execute(f"PRAGMA busy_timeout={config.DB_BUSY_TIMEOUT_MS};")

    try:
        yield db
    finally:
        await db.close()


async def init_db() -> None:
    """Создает таблицу predictions при старте приложения."""

    async with get_db() as db:
        # WAL улучшает совместную работу чтения и записи
        await db.execute("PRAGMA journal_mode=WAL;")
        await db.execute("PRAGMA synchronous=NORMAL;")

        await db.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                request_id TEXT NOT NULL,
                farm_id TEXT NOT NULL,
                risk_score REAL NOT NULL,
                risk_level TEXT NOT NULL,
                recommendation TEXT NOT NULL,
                model_version TEXT NOT NULL,
                source TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            """
        )

        await db.execute(
            "CREATE INDEX IF NOT EXISTS idx_predictions_request_id "
            "ON predictions(request_id);"
        )

        await db.execute(
            "CREATE INDEX IF NOT EXISTS idx_predictions_risk_level "
            "ON predictions(risk_level);"
        )

        await db.commit()

    logger.info("SQLite initialized | path=%s", config.DATABASE_PATH)


# ------------------------------------------------------------
# ЗАПИСЬ
# ------------------------------------------------------------

async def save_prediction(result: dict, source: str) -> None:
    """
    Сохраняет один прогноз.

    request_id не PRIMARY KEY: один batch-запрос порождает
    несколько прогнозов с одним request_id.
    """

    try:
        async with get_db() as db:
            await db.execute(
                INSERT_SQL,
                (
                    result["request_id"],
                    result["farm_id"],
                    result["risk_score"],
                    result["risk_level"],
                    result["recommendation"],
                    result["model_version"],
                    source,
                )
            )
            await db.commit()

    except aiosqlite.Error as exc:
        logger.exception(
            "Database write error | request_id=%s",
            result.get("request_id")
        )
        raise storage_unavailable() from exc


async def save_predictions_batch(
    request_id: str,
    items: List[dict]
) -> None:
    """Сохраняет весь batch через executemany() и один COMMIT."""

    rows = [
        (
            request_id,
            item["farm_id"],
            item["risk_score"],
            item["risk_level"],
            item["recommendation"],
            item["model_version"],
            "batch",
        )
        for item in items
    ]

    try:
        async with get_db() as db:
            await db.executemany(INSERT_SQL, rows)
            await db.commit()

    except aiosqlite.Error as exc:
        logger.exception(
            "Batch database write error | request_id=%s",
            request_id
        )
        raise storage_unavailable() from exc


# ------------------------------------------------------------
# ЧТЕНИЕ
# ------------------------------------------------------------

async def list_predictions(
    limit: int,
    risk_level: Optional[str] = None
) -> List[dict]:
    """История прогнозов из SQLite (новые сверху)."""

    query = f"SELECT {SELECT_COLUMNS} FROM predictions"
    params: list = []

    if risk_level is not None:
        query += " WHERE risk_level = ?"
        params.append(risk_level)

    query += " ORDER BY id DESC LIMIT ?;"
    params.append(limit)

    try:
        async with get_db() as db:
            cursor = await db.execute(query, params)
            rows = await cursor.fetchall()
            await cursor.close()

        return [dict(row) for row in rows]

    except aiosqlite.Error as exc:
        logger.exception("Database read error")
        raise storage_unavailable() from exc


async def get_predictions_by_request_id(request_id: str) -> List[dict]:
    """
    Все прогнозы одного HTTP request_id.

    Для /predict и /predict-queued - одна строка,
    для /batch-predict - несколько.
    """

    try:
        async with get_db() as db:
            cursor = await db.execute(
                f"SELECT {SELECT_COLUMNS} FROM predictions "
                "WHERE request_id = ? ORDER BY id;",
                (request_id,)
            )
            rows = await cursor.fetchall()
            await cursor.close()

        return [dict(row) for row in rows]

    except aiosqlite.Error as exc:
        logger.exception(
            "Database read error | request_id=%s",
            request_id
        )
        raise storage_unavailable() from exc


async def get_predictions_by_weekday_page(
    weekday_number: str,
    page: int,
    per_page: int
) -> tuple[List[dict], int, int]:
    """
    Одна страница прогнозов для выбранного дня недели
    (offset-пагинация: LIMIT / OFFSET + COUNT(*)).

        page=1, per_page=5 -> LIMIT 5 OFFSET 0
        page=2, per_page=5 -> LIMIT 5 OFFSET 5
    """

    offset = (page - 1) * per_page

    try:
        async with get_db() as db:
            cursor = await db.execute(
                f"SELECT {SELECT_COLUMNS} FROM predictions "
                "WHERE strftime('%w', created_at) = ? "
                "ORDER BY id DESC LIMIT ? OFFSET ?;",
                (weekday_number, per_page, offset)
            )
            rows = await cursor.fetchall()
            await cursor.close()

            cursor = await db.execute(
                "SELECT COUNT(*) FROM predictions "
                "WHERE strftime('%w', created_at) = ?;",
                (weekday_number,)
            )
            count_row = await cursor.fetchone()
            await cursor.close()

        total_items = count_row[0]
        total_pages = (
            math.ceil(total_items / per_page)
            if total_items > 0
            else 0
        )

        return [dict(row) for row in rows], total_items, total_pages

    except aiosqlite.Error as exc:
        logger.exception(
            "Database read error | weekday=%s",
            weekday_number
        )
        raise storage_unavailable() from exc


# ------------------------------------------------------------
# ДЕМО-ДАННЫЕ ДЛЯ ПАГИНАЦИИ
# ------------------------------------------------------------

async def seed_weekday_predictions(count: int = 70) -> int:
    """
    Создает демонстрационные прогнозы за последние 21 день,
    чтобы в базе были записи для всех дней недели.
    """

    rows = []

    for i in range(count):
        risk_score = round(min(0.1 + (i % 9) * 0.1, 1.0), 2)
        risk_level = get_risk_level(risk_score)

        rows.append(
            (
                str(uuid.uuid4()),
                f"DEMO-{i + 1:03d}",
                risk_score,
                risk_level,
                get_recommendation(risk_level),
                model.MODEL_VERSION,
                "seed",
                f"-{i % 21} days",
            )
        )

    try:
        async with get_db() as db:
            await db.executemany(
                """
                INSERT INTO predictions (
                    request_id, farm_id, risk_score, risk_level,
                    recommendation, model_version, source, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, datetime('now', ?));
                """,
                rows
            )
            await db.commit()

        return count

    except aiosqlite.Error as exc:
        logger.exception("Demo seed database error")
        raise storage_unavailable() from exc
