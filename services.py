# Постпроцессинг и бизнес-логика

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
