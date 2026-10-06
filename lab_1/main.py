# ПРАКТИЧЕСКОЕ ЗАНЯТИЕ № 1
# Разработка REST API для AI-сервиса на FastAPI
#
# Структура проекта:
#   main.py     - приложение FastAPI, middleware, endpoints
#   schemas.py  - Pydantic-модели запросов и ответов
#   model.py    - модель оценки риска и ее параметры
#   services.py - постпроцессинг и бизнес-логика
#   storage.py  - хранилище результатов

import time
import uuid
import logging
from typing import Optional, List

from fastapi import (
    FastAPI,
    HTTPException,
    Query,
    Request,
    status
)

import model
import storage
from schemas import (
    FarmRequest,
    PredictionResponse,
    HealthResponse,
    ModelInfoResponse
)
from services import (
    ALLOWED_LEVELS,
    ALLOWED_REGIONS,
    get_risk_level,
    get_recommendation,
    is_region_allowed
)


# ------------------------------------------------------------
# НАСТРОЙКА ЛОГИРОВАНИЯ
# ------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)


# ------------------------------------------------------------
# FASTAPI-ПРИЛОЖЕНИЕ
# ------------------------------------------------------------

app = FastAPI(
    title="Agro Scoring API",
    description=(
        "REST API для оценки риска "
        "сельскохозяйственных предприятий."
    ),
    version="1.0.0"
)


# ------------------------------------------------------------
# MIDDLEWARE
# Измеряем время обработки каждого HTTP-запроса.
# ------------------------------------------------------------

@app.middleware("http")
async def add_process_time(
    request: Request,
    call_next
):
    """
    Middleware выполняется для каждого HTTP-запроса.
    Добавляет в HTTP-ответ заголовок X-Process-Time
    со временем обработки запроса.
    """

    start_time = time.perf_counter()

    response = await call_next(request)

    process_time = time.perf_counter() - start_time

    response.headers["X-Process-Time"] = str(
        round(process_time, 6)
    )

    return response


# ------------------------------------------------------------
# GET /health
# ------------------------------------------------------------

@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Проверка состояния API",
    description=(
        "Используется для проверки того, "
        "что REST API запущен и отвечает."
    )
)
def health():
    """
    Проверка работоспособности API.
    """

    return {
        "status": "ok"
    }


# ------------------------------------------------------------
# GET /model-info
# ------------------------------------------------------------

@app.get(
    "/model-info",
    response_model=ModelInfoResponse,
    summary="Информация о модели",
    description=(
        "Возвращает название, версию, тип "
        "и текущее состояние модели."
    )
)
def model_info():
    """
    Возвращает информацию об используемой ML-модели.
    """

    model_status = (
        "ready"
        if model.MODEL_READY
        else "unavailable"
    )

    return {
        "model_name": model.MODEL_NAME,
        "model_version": model.MODEL_VERSION,
        "model_type": model.MODEL_TYPE,
        "status": model_status
    }


# ------------------------------------------------------------
# POST /predict
# ------------------------------------------------------------

@app.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Оценить риск хозяйства",
    description=(
        "Принимает характеристики хозяйства, "
        "выполняет валидацию, инференс модели "
        "и возвращает оценку риска."
    )
)
def predict(request: FarmRequest):
    """
    Основной endpoint сервиса.
    """

    # 1. Проверяем состояние ML-модели

    if not model.MODEL_READY:

        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is temporarily unavailable"
        )

    # 2. Бизнес-валидация
    #
    # Pydantic уже проверил типы, обязательность и диапазоны.
    # Осталось проверить бизнес-правила (разрешенные регионы).

    if not is_region_allowed(request.region):

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Unknown region: {request.region}. "
                f"Allowed regions: "
                f"{sorted(ALLOWED_REGIONS)}"
            )
        )

    # 3. Логирование входящего запроса

    logger.info(
        "Prediction request received | farm_id=%s",
        request.farm_id
    )

    # 4. ИНФЕРЕНС

    score = model.calculate_risk(request)

    # 5. ПОСТПРОЦЕССИНГ

    level = get_risk_level(score)

    recommendation = get_recommendation(level)

    # 6. СОЗДАЕМ УНИКАЛЬНЫЙ ID ЗАПРОСА

    request_id = str(uuid.uuid4())

    # 7. ФОРМИРУЕМ РЕЗУЛЬТАТ

    result = {
        "request_id": request_id,
        "farm_id": request.farm_id,
        "risk_score": score,
        "risk_level": level,
        "recommendation": recommendation,
        "model_version": model.MODEL_VERSION
    }

    # 8. СОХРАНЯЕМ РЕЗУЛЬТАТ

    storage.save_prediction(request_id, result)

    # 9. ЛОГИРОВАНИЕ РЕЗУЛЬТАТА

    logger.info(
        "Prediction completed | "
        "request_id=%s | "
        "farm_id=%s | "
        "risk_score=%s | "
        "risk_level=%s",
        request_id,
        request.farm_id,
        score,
        level
    )

    # 10. ВОЗВРАЩАЕМ HTTP-ОТВЕТ

    return result


# ------------------------------------------------------------
# GET /predictions
#
# этот endpoint должен быть расположен ДО
# /predictions/{request_id}
# ------------------------------------------------------------

@app.get(
    "/predictions",
    response_model=List[PredictionResponse],
    summary="Получить список прогнозов",
    description=(
        "Возвращает список выполненных прогнозов. "
        "Поддерживает ограничение количества результатов "
        "и фильтрацию по уровню риска."
    )
)
def get_predictions(

    limit: int = Query(
        default=10,
        ge=1,
        le=100,
        description="Максимальное количество результатов"
    ),

    risk_level: Optional[str] = Query(
        default=None,
        description=(
            "Фильтр по категории риска: "
            "low, medium или high"
        )
    )

):
    """
    Получение списка прогнозов.
    """

    # Проверяем корректность risk_level

    if risk_level is not None and risk_level not in ALLOWED_LEVELS:

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "risk_level must be "
                "'low', 'medium' or 'high'"
            )
        )

    # Фильтрация и ограничение количества результатов

    return storage.list_predictions(limit, risk_level)


# ------------------------------------------------------------
# GET /predictions/{request_id}
# ------------------------------------------------------------

@app.get(
    "/predictions/{request_id}",
    response_model=PredictionResponse,
    summary="Получить прогноз по request_id",
    description=(
        "Возвращает сохраненный прогноз "
        "по его уникальному идентификатору."
    )
)
def get_prediction(request_id: str):
    """
    Получение одного прогноза по request_id.
    """

    prediction = storage.get_prediction_by_id(request_id)

    # Проверяем наличие прогноза

    if prediction is None:

        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Prediction not found"
        )

    # Возвращаем прогноз

    return prediction


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
