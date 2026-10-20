# Модель оценки риска и ее параметры

import asyncio
from typing import List

import config
from schemas import FarmRequest

MODEL_NAME = "agro-risk-model"
MODEL_VERSION = "1.0"
MODEL_TYPE = "risk-scoring"

# Имитируем состояние модели.
# Если установить False, endpoints инференса вернут HTTP 503.
MODEL_READY = True


def calculate_risk(data: FarmRequest) -> float:
    """
    Простая логика оценки риска по финансовым показателям
    и отраслевым факторам. Позже будет заменена на ML-модель.
    """

    score = 0.1

    # Длительная просрочка увеличивает риск
    if data.payment_delay_days > 30:
        score += 0.3

    # Наличие предыдущих дефолтов увеличивает риск
    if data.previous_defaults > 0:
        score += 0.3

    # Большая задолженность увеличивает риск
    if data.debt > 5_000_000:
        score += 0.2

    # Малое количество осадков условно увеличивает аграрный риск
    if data.precipitation_mm < 100:
        score += 0.1

    # Ограничиваем значение диапазоном от 0 до 1
    score = min(score, 1.0)

    return round(score, 2)


# ------------------------------------------------------------
# ИМИТАЦИЯ INFERENCE ENGINE
#
# Один lock имитирует один экземпляр модели / один
# вычислительный ресурс: два вызова одновременно не идут.
# ------------------------------------------------------------

model_lock = asyncio.Lock()


async def model_predict_single(data: FarmRequest) -> float:
    """Одиночный вызов модели."""

    async with model_lock:
        await asyncio.sleep(
            config.MODEL_FIXED_OVERHEAD_SECONDS
            + config.MODEL_PER_ITEM_SECONDS
        )

        return calculate_risk(data)


async def model_predict_batch(items: List[FarmRequest]) -> List[float]:
    """
    Один пакетный вызов модели: фиксированный overhead
    оплачивается один раз на весь пакет.
    """

    async with model_lock:
        await asyncio.sleep(
            config.MODEL_FIXED_OVERHEAD_SECONDS
            + config.MODEL_PER_ITEM_SECONDS * len(items)
        )

        return [calculate_risk(item) for item in items]
