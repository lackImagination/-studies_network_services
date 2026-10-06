# Модель оценки риска и ее параметры

from schemas import FarmRequest

MODEL_NAME = "agro-risk-model"
MODEL_VERSION = "1.0"
MODEL_TYPE = "risk-scoring"

# Имитируем состояние модели.
# Если установить False, /predict будет возвращать HTTP 503.
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
