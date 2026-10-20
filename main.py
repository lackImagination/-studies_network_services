# Структура проекта:
#   main.py     - приложение FastAPI, middleware, обработчики ошибок, endpoints
#   config.py   - настройки: batching, очередь, таймауты, SQLite, LLM
#   schemas.py  - Pydantic-модели запросов и ответов
#   model.py    - модель оценки риска, одиночный и пакетный инференс
#   services.py - постпроцессинг и бизнес-логика
#   storage.py  - SQLite через aiosqlite
#   batching.py - очередь asyncio.Queue и dynamic batching worker
#   llm.py      - асинхронный клиент внешней LLM (Vireonix)

import asyncio
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import List, Optional

import httpx
from fastapi import (
    FastAPI,
    HTTPException,
    Query,
    Request,
    status
)
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

import batching
import config
import model
import storage
from llm import build_scoring_explanation_prompt, call_vireonix_llm
from schemas import (
    BatchPredictionResponse,
    BatchRequest,
    FarmRequest,
    HealthResponse,
    LLMChatRequest,
    LLMChatResponse,
    ModelInfoResponse,
    PaginatedPredictionsResponse,
    PredictionResponse,
    PredictWithLLMResponse
)
from services import (
    ALLOWED_LEVELS,
    WEEKDAY_MAP,
    get_recommendation,
    get_risk_level,
    make_prediction_result,
    validate_business_rules
)


# ------------------------------------------------------------
# ЛОГИРОВАНИЕ
# ------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)


# ------------------------------------------------------------
# LIFESPAN: при старте - SQLite, HTTP-клиент LLM, batch worker;
# при остановке - все закрываем
# ------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    await storage.init_db()

    app.state.llm_client = httpx.AsyncClient(
        timeout=httpx.Timeout(config.LLM_TIMEOUT_SECONDS)
    )

    worker = asyncio.create_task(batching.batch_worker())
    app.state.batch_worker = worker

    yield

    worker.cancel()

    try:
        await worker
    except asyncio.CancelledError:
        pass

    await app.state.llm_client.aclose()


# ------------------------------------------------------------
# FASTAPI-ПРИЛОЖЕНИЕ
# ------------------------------------------------------------

app = FastAPI(
    title="Agro Scoring API",
    description=(
        "Agro Scoring API: request_id, SQLite/aiosqlite, "
        "single inference, batching и интеграция с Vireonix LLM."
    ),
    version="2.0.0",
    lifespan=lifespan
)


# ------------------------------------------------------------
# MIDDLEWARE: request_id + время обработки
# ------------------------------------------------------------

@app.middleware("http")
async def request_context_middleware(request: Request, call_next):
    """
    request_id берется из X-Request-ID или создается как UUID,
    сохраняется в request.state и возвращается клиенту в заголовке.
    Время обработки возвращается в X-Process-Time.
    """

    start_time = time.perf_counter()

    request_id = request.headers.get("X-Request-ID")

    if not request_id:
        request_id = str(uuid.uuid4())

    request.state.request_id = request_id

    logger.info(
        "Request started | request_id=%s | method=%s | path=%s",
        request_id,
        request.method,
        request.url.path
    )

    try:
        response = await call_next(request)

    except Exception:
        logger.exception("Unhandled error | request_id=%s", request_id)

        response = JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "detail": "Internal server error",
                "request_id": request_id
            }
        )

    process_time = time.perf_counter() - start_time

    response.headers["X-Request-ID"] = request_id
    response.headers["X-Process-Time"] = str(round(process_time, 6))

    logger.info(
        "Request completed | request_id=%s | status=%s | time=%.4f",
        request_id,
        response.status_code,
        process_time
    )

    return response


# ------------------------------------------------------------
# ОБРАБОТЧИКИ ОШИБОК
# Любой ответ с ошибкой содержит request_id.
# ------------------------------------------------------------

@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(
    request: Request,
    exc: StarletteHTTPException
):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "detail": exc.detail,
            "request_id": getattr(request.state, "request_id", None)
        },
        headers=getattr(exc, "headers", None)
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError
):
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "detail": jsonable_encoder(exc.errors()),
            "request_id": getattr(request.state, "request_id", None)
        }
    )


# ------------------------------------------------------------
# GET /health
# ------------------------------------------------------------

@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Проверка состояния API"
)
async def health():
    return {"status": "ok"}


# ------------------------------------------------------------
# GET /model-info
# ------------------------------------------------------------

@app.get(
    "/model-info",
    response_model=ModelInfoResponse,
    summary="Информация о модели"
)
async def model_info():
    return {
        "model_name": model.MODEL_NAME,
        "model_version": model.MODEL_VERSION,
        "model_type": model.MODEL_TYPE,
        "status": "ready" if model.MODEL_READY else "unavailable"
    }


# ------------------------------------------------------------
# Общие шаги для endpoints инференса
# ------------------------------------------------------------

def ensure_model_ready() -> None:
    if not model.MODEL_READY:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is unavailable"
        )


async def run_single_inference(data: FarmRequest) -> float:
    try:
        return await asyncio.wait_for(
            model.model_predict_single(data),
            timeout=config.INFERENCE_TIMEOUT_SECONDS
        )

    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="Inference timeout"
        )


# ------------------------------------------------------------
# POST /predict - одиночный inference
# ------------------------------------------------------------

@app.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Одиночный inference"
)
async def predict(data: FarmRequest, request: Request):
    """Один HTTP-запрос -> один inference -> одна запись в SQLite."""

    ensure_model_ready()
    validate_business_rules(data)

    request_id = request.state.request_id

    score = await run_single_inference(data)

    result = make_prediction_result(data, score, request_id)

    # HTTP 200 возвращается только после успешного COMMIT
    await storage.save_prediction(result, source="single")

    return result


# ------------------------------------------------------------
# POST /batch-predict - явный пакетный inference
# ------------------------------------------------------------

@app.post(
    "/batch-predict",
    response_model=BatchPredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Явный пакетный inference"
)
async def batch_predict(batch: BatchRequest, request: Request):
    """Один вызов модели на весь пакет и одна транзакция в БД."""

    ensure_model_ready()

    for item in batch.items:
        validate_business_rules(item)

    request_id = request.state.request_id

    start_time = time.perf_counter()

    try:
        scores = await asyncio.wait_for(
            model.model_predict_batch(batch.items),
            timeout=config.INFERENCE_TIMEOUT_SECONDS
        )

    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="Batch inference timeout"
        )

    response_items = []

    for data, score in zip(batch.items, scores):
        level = get_risk_level(score)

        response_items.append(
            {
                "farm_id": data.farm_id,
                "risk_score": score,
                "risk_level": level,
                "recommendation": get_recommendation(level),
                "model_version": model.MODEL_VERSION
            }
        )

    await storage.save_predictions_batch(
        request_id=request_id,
        items=response_items
    )

    processing_time_ms = (time.perf_counter() - start_time) * 1000

    return {
        "request_id": request_id,
        "batch_size": len(batch.items),
        "processing_time_ms": round(processing_time_ms, 2),
        "items": response_items
    }


# ------------------------------------------------------------
# POST /predict-queued - очередь и dynamic batching
# ------------------------------------------------------------

@app.post(
    "/predict-queued",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Inference через очередь и dynamic batching"
)
async def predict_queued(data: FarmRequest, request: Request):
    """Запрос ставится в очередь, worker объединяет их в batch."""

    ensure_model_ready()
    validate_business_rules(data)

    request_id = request.state.request_id

    future = asyncio.get_running_loop().create_future()

    task = batching.InferenceTask(
        data=data,
        request_id=request_id,
        future=future
    )

    # Backpressure: put_nowait не ждет, а сразу выбрасывает QueueFull,
    # если очередь заполнена. Клиент получает 429 и повторяет запрос позже.
    try:
        batching.inference_queue.put_nowait(task)

    except asyncio.QueueFull:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Inference queue is full. Try again later.",
            headers={"Retry-After": "1"}
        )

    logger.info(
        "Request queued | request_id=%s | queue_size=%s",
        request_id,
        batching.inference_queue.qsize()
    )

    try:
        score = await asyncio.wait_for(
            future,
            timeout=config.INFERENCE_TIMEOUT_SECONDS
        )

    except asyncio.TimeoutError:
        if not future.done():
            future.cancel()

        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="Inference timeout"
        )

    result = make_prediction_result(data, score, request_id)

    await storage.save_prediction(result, source="queued")

    return result


# ------------------------------------------------------------
# GET /queue-status
# ------------------------------------------------------------

@app.get(
    "/queue-status",
    summary="Состояние очереди и настройки batching"
)
async def queue_status():
    return {
        "queue_size": batching.inference_queue.qsize(),
        "queue_max_size": config.QUEUE_MAX_SIZE,
        "max_batch_size": config.MAX_BATCH_SIZE,
        "max_batch_wait_ms": config.MAX_BATCH_WAIT_MS,
        "inference_timeout_seconds": config.INFERENCE_TIMEOUT_SECONDS
    }


# ------------------------------------------------------------
# POST /llm/chat - прямой запрос к LLM
# ------------------------------------------------------------

@app.post(
    "/llm/chat",
    response_model=LLMChatResponse,
    status_code=status.HTTP_200_OK,
    summary="Прямой учебный запрос к Vireonix LLM",
    description=(
        "Отправляет prompt в OpenAI-совместимый Vireonix API. "
        "API-ключ не используется. "
        "Не передавайте реальные конфиденциальные данные."
    )
)
async def llm_chat(data: LLMChatRequest, request: Request):
    return await call_vireonix_llm(
        client=request.app.state.llm_client,
        prompt=data.prompt,
        request_id=request.state.request_id,
        model=data.model,
        temperature=data.temperature,
        max_tokens=data.max_tokens
    )


# ------------------------------------------------------------
# POST /predict-with-llm - скоринг + объяснение от LLM
# ------------------------------------------------------------

@app.post(
    "/predict-with-llm",
    response_model=PredictWithLLMResponse,
    status_code=status.HTTP_200_OK,
    summary="Агроскоринг с текстовым объяснением LLM",
    description=(
        "Сначала выполняется детерминированный скоринг, результат "
        "сохраняется в SQLite. Затем LLM формирует только текстовое "
        "объяснение. LLM не определяет risk_score и risk_level."
    )
)
async def predict_with_llm(data: FarmRequest, request: Request):
    ensure_model_ready()
    validate_business_rules(data)

    request_id = request.state.request_id

    score = await run_single_inference(data)

    prediction = make_prediction_result(data, score, request_id)

    # Сначала сохраняем детерминированный прогноз:
    # если LLM недоступна, результат скоринга не теряется
    await storage.save_prediction(prediction, source="single+llm")

    # В публичный LLM уходит только результат скоринга,
    # без farm_id и исходных финансовых признаков
    llm_result = await call_vireonix_llm(
        client=request.app.state.llm_client,
        prompt=build_scoring_explanation_prompt(prediction),
        request_id=request_id,
        # Для пересказа готового результата нужна максимальная
        # детерминированность, а не творчество
        temperature=0.0
    )

    return {
        "prediction": prediction,
        "llm": llm_result
    }


# ------------------------------------------------------------
# GET /predictions
#
# Маршруты с фиксированным путем (/predictions/by-weekday)
# должны быть ДО /predictions/{request_id}
# ------------------------------------------------------------

@app.get(
    "/predictions",
    response_model=List[PredictionResponse],
    summary="История прогнозов из SQLite"
)
async def get_predictions(
    limit: int = Query(default=10, ge=1, le=100),
    risk_level: Optional[str] = Query(
        default=None,
        description="Фильтр: low, medium или high"
    )
):
    if risk_level is not None and risk_level not in ALLOWED_LEVELS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="risk_level must be 'low', 'medium' or 'high'"
        )

    return await storage.list_predictions(limit, risk_level)


# ------------------------------------------------------------
# GET /predictions/by-weekday - пагинация + фильтр по дню недели
# ------------------------------------------------------------

@app.get(
    "/predictions/by-weekday",
    response_model=PaginatedPredictionsResponse,
    summary="Прогнозы по дню недели с пагинацией"
)
async def get_predictions_by_weekday(
    request: Request,
    weekday: str = Query(
        ...,
        description=(
            "monday, tuesday, wednesday, thursday, "
            "friday, saturday, sunday"
        )
    ),
    page: int = Query(default=1, ge=1, description="Номер страницы"),
    per_page: int = Query(
        default=5,
        ge=1,
        le=100,
        description="Записей на странице"
    )
):
    weekday = weekday.lower()

    if weekday not in WEEKDAY_MAP:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "weekday must be one of: monday, tuesday, wednesday, "
                "thursday, friday, saturday, sunday"
            )
        )

    data, total_items, total_pages = (
        await storage.get_predictions_by_weekday_page(
            weekday_number=WEEKDAY_MAP[weekday],
            page=page,
            per_page=per_page
        )
    )

    # Записи есть, но номер страницы слишком большой
    if total_pages > 0 and page > total_pages:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Page {page} does not exist. Total pages: {total_pages}"
        )

    # Записей нет вообще - разрешена только page=1
    if total_pages == 0 and page > 1:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No data for this weekday"
        )

    next_page = None

    if page < total_pages:
        next_page = str(
            request.url.include_query_params(
                weekday=weekday,
                page=page + 1,
                per_page=per_page
            )
        )

    prev_page = None

    if page > 1 and total_pages > 0:
        prev_page = str(
            request.url.include_query_params(
                weekday=weekday,
                page=page - 1,
                per_page=per_page
            )
        )

    return {
        "data": data,
        "meta": {
            "weekday": weekday,
            "current_page": page,
            "per_page": per_page,
            "total_pages": total_pages,
            "total_items": total_items,
            "next_page": next_page,
            "prev_page": prev_page
        }
    }


# ------------------------------------------------------------
# POST /demo/seed-weekdays - демо-данные за разные дни
# ------------------------------------------------------------

@app.post(
    "/demo/seed-weekdays",
    summary="Создать демонстрационные прогнозы за разные дни"
)
async def seed_demo_weekdays(
    count: int = Query(default=70, ge=7, le=1000)
):
    """Данные за последние 21 день для проверки пагинации."""

    created = await storage.seed_weekday_predictions(count=count)

    return {
        "status": "ok",
        "created": created,
        "message": (
            "Demo predictions created. "
            "Use /predictions/by-weekday for pagination."
        )
    }


# ------------------------------------------------------------
# GET /requests/{request_id}/predictions - все прогнозы запроса
# ------------------------------------------------------------

@app.get(
    "/requests/{request_id}/predictions",
    response_model=List[PredictionResponse],
    summary="Все прогнозы, созданные одним HTTP-запросом"
)
async def get_request_predictions(request_id: str):
    results = await storage.get_predictions_by_request_id(request_id)

    if not results:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Prediction not found"
        )

    return results


# ------------------------------------------------------------
# GET /predictions/{request_id}
# ------------------------------------------------------------

@app.get(
    "/predictions/{request_id}",
    response_model=PredictionResponse,
    summary="Получить одиночный прогноз по request_id"
)
async def get_prediction(request_id: str):
    """Для batch (несколько строк) - 409, см. /requests/{id}/predictions."""

    results = await storage.get_predictions_by_request_id(request_id)

    if not results:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Prediction not found"
        )

    if len(results) > 1:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                "This request_id contains multiple predictions. "
                f"Use /requests/{request_id}/predictions"
            )
        )

    return results[0]


@app.get("/")
async def root():
    return {
        "message": "Agro Scoring API is running",
        "docs": "/docs",
        "health": "/health"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
