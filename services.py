# Постпроцессинг и бизнес-логика

from fastapi import HTTPException, status

import model
from schemas import FarmRequest

# Разрешенные регионы (пример бизнес-справочника)
ALLOWED_REGIONS = {
    "Krasnodar",
    "Rostov",
    "Stavropol"
}

# Допустимые категории риска
ALLOWED_LEVELS = {
    "low",
    "medium",
    "high"
}

# SQLite strftime('%w', ...) возвращает:
# 0 = Sunday, 1 = Monday, ..., 6 = Saturday.
WEEKDAY_MAP = {
    "sunday": "0",
    "monday": "1",
    "tuesday": "2",
    "wednesday": "3",
    "thursday": "4",
    "friday": "5",
    "saturday": "6"
}


def get_risk_level(score: float) -> str:
    """Преобразует score в категорию риска."""

    if score < 0.3:
        return "low"

    if score < 0.7:
        return "medium"

    return "high"


def get_recommendation(level: str) -> str:
    """Возвращает рекомендацию в зависимости от категории риска."""

    if level == "low":
        return "Стандартное рассмотрение"

    if level == "medium":
        return "Требуется дополнительная проверка"

    return "Высокий риск. Требуется ручное рассмотрение"


def is_region_allowed(region: str) -> bool:
    """Бизнес-правило: регион должен быть из справочника."""

    return region in ALLOWED_REGIONS


def validate_business_rules(data: FarmRequest) -> None:
    """
    Pydantic проверяет типы, обязательность и диапазоны.
    Здесь проверяются правила предметной области.
    """

    if not is_region_allowed(data.region):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Unknown region: {data.region}. "
                f"Allowed regions: {sorted(ALLOWED_REGIONS)}"
            )
        )


def make_prediction_result(
    data: FarmRequest,
    score: float,
    request_id: str
) -> dict:
    """Собирает результат прогноза: постпроцессинг score."""

    level = get_risk_level(score)

    return {
        "request_id": request_id,
        "farm_id": data.farm_id,
        "risk_score": score,
        "risk_level": level,
        "recommendation": get_recommendation(level),
        "model_version": model.MODEL_VERSION
    }
