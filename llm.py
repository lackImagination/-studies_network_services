# Интеграция с внешней LLM (Vireonix, OpenAI-совместимый API)

import logging
import time

import httpx
from fastapi import HTTPException, status

import config

logger = logging.getLogger(__name__)


async def call_vireonix_llm(
    client: httpx.AsyncClient,
    prompt: str,
    request_id: str,
    model: str = config.LLM_MODEL,
    temperature: float = config.LLM_DEFAULT_TEMPERATURE,
    max_tokens: int = config.LLM_DEFAULT_MAX_TOKENS
) -> dict:
    """
    Асинхронный запрос POST /v1/chat/completions.

    Используется httpx.AsyncClient + await, а не requests.post():
    блокирующий HTTP-вызов остановил бы весь event loop.

    Ошибки внешней системы переводятся в наши коды:
    timeout -> 504, нет соединения -> 503,
    ошибка провайдера / битый ответ -> 502.
    """

    url = f"{config.LLM_BASE_URL}/chat/completions"

    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Ты помощник сервиса агроскоринга. "
                    "Отвечай кратко и понятно. "
                    "Не изменяй переданный risk_score и risk_level. "
                    "Не придумывай причины, которых нет во входных данных."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": temperature,
        "max_tokens": max_tokens
    }

    started = time.perf_counter()

    try:
        response = await client.post(
            url,
            headers={"Content-Type": "application/json"},
            json=payload
        )

    except httpx.TimeoutException as exc:
        logger.exception("LLM timeout | request_id=%s", request_id)
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="LLM request timeout"
        ) from exc

    except httpx.RequestError as exc:
        logger.exception("LLM connection error | request_id=%s", request_id)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="LLM service is unavailable"
        ) from exc

    latency_ms = (time.perf_counter() - started) * 1000

    if response.status_code != status.HTTP_200_OK:
        logger.error(
            "LLM upstream error | request_id=%s | status=%s | body=%s",
            request_id,
            response.status_code,
            response.text[:500]
        )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=(
                "LLM provider returned an error "
                f"(HTTP {response.status_code})"
            )
        )

    try:
        data = response.json()
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="LLM provider returned invalid JSON"
        ) from exc

    choices = data.get("choices", [])

    if not choices:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="LLM response does not contain choices"
        )

    choice = choices[0]
    message = choice.get("message", {})
    content = message.get("content")

    if not content:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="LLM response does not contain message content"
        )

    return {
        "request_id": request_id,
        "requested_model": model,
        "returned_model": data.get("model"),
        "answer": content,
        "reasoning": message.get("reasoning_content"),
        "finish_reason": choice.get("finish_reason"),
        "usage": data.get("usage"),
        "latency_ms": round(latency_ms, 2)
    }


def build_scoring_explanation_prompt(prediction: dict) -> str:
    """
    Промпт строится только из результата скорингового алгоритма.

    farm_id и исходные финансовые признаки в публичный
    LLM gateway не отправляются.
    """

    level_ru = {
        "low": "низкий",
        "medium": "средний",
        "high": "высокий"
    }.get(prediction["risk_level"], prediction["risk_level"])

    return (
        "Ниже готовый результат алгоритмического агроскоринга. "
        "Перескажи его оператору банка, ничего не меняя.\n\n"
        "РЕЗУЛЬТАТ (эти значения нельзя менять или заменять другими):\n"
        f"- оценка риска (risk_score): {prediction['risk_score']} "
        "по шкале от 0 до 1\n"
        f"- уровень риска (risk_level): {prediction['risk_level']} "
        f"({level_ru})\n"
        f"- базовая рекомендация: {prediction['recommendation']}\n"
        f"- версия скоринговой модели: {prediction['model_version']}\n\n"
        "Требования к ответу:\n"
        "1. 2-4 предложения.\n"
        f"2. Обязательно назови оценку {prediction['risk_score']} "
        f"и уровень риска: {level_ru}.\n"
        "3. Не придумывай дополнительные факторы риска и другие числа.\n"
        "4. Укажи, что решение требует применения правил банка "
        "и при необходимости проверки оператором."
    )
