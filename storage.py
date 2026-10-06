# Временное хранилище результатов в оперативной памяти.
# Позже можно заменить на PostgreSQL или SQLite,
# не меняя остальной код (интерфейс функций сохранится).

from typing import List, Optional

predictions = {}


def save_prediction(request_id: str, result: dict) -> None:
    """Сохраняет результат прогноза."""

    predictions[request_id] = result


def get_prediction_by_id(request_id: str) -> Optional[dict]:
    """Возвращает прогноз по id или None, если его нет."""

    return predictions.get(request_id)


def list_predictions(
    limit: int,
    risk_level: Optional[str] = None
) -> List[dict]:
    """Возвращает список прогнозов с фильтром и ограничением."""

    values = list(predictions.values())

    if risk_level is not None:
        values = [
            item
            for item in values
            if item["risk_level"] == risk_level
        ]

    return values[:limit]
