# Agro Scoring API

REST API для оценки риска сельскохозяйственных предприятий на FastAPI.

- Практическое занятие №1: базовый API (`/predict`, хранилище в памяти).
- Лабораторная работа №2: сквозной `request_id`, middleware, SQLite (`aiosqlite`),
  пакетный инференс, очередь и dynamic batching, backpressure, интеграция с LLM.

## Запуск

```
pip install -r requirements.txt
python main.py
```

Swagger: http://127.0.0.1:8000/docs

## Структура

| Файл | Назначение |
|------|-----------|
| `main.py` | приложение FastAPI, middleware, обработчики ошибок, endpoints |
| `config.py` | настройки: batching, очередь, таймауты, SQLite, LLM |
| `schemas.py` | Pydantic-модели запросов и ответов |
| `model.py` | модель оценки риска, одиночный и пакетный инференс |
| `services.py` | постпроцессинг и бизнес-правила |
| `storage.py` | SQLite через aiosqlite |
| `batching.py` | очередь `asyncio.Queue` и worker dynamic batching |
| `llm.py` | асинхронный клиент внешней LLM (Vireonix) |

## Endpoints

- GET /health
- GET /model-info
- POST /predict - одиночный инференс
- POST /batch-predict - явный пакетный инференс (до 8 объектов)
- POST /predict-queued - очередь и dynamic batching
- GET /queue-status - состояние очереди
- GET /predictions - история из SQLite
- GET /predictions/by-weekday - пагинация по дню недели
- GET /predictions/{request_id}
- GET /requests/{request_id}/predictions - все прогнозы одного запроса
- POST /llm/chat - прямой запрос к LLM
- POST /predict-with-llm - скоринг + текстовое объяснение от LLM
- POST /demo/seed-weekdays - демо-данные за разные дни

## Настройки через переменные окружения

- `INFERENCE_TIMEOUT_SECONDS` - таймаут инференса (по умолчанию 2.0)
- `DATABASE_PATH` - путь к файлу SQLite
- `LLM_BASE_URL` - адрес OpenAI-совместимого LLM API
- `QUEUE_MAX_SIZE` - размер очереди (по умолчанию 100)
- `MODEL_FIXED_OVERHEAD_SECONDS` - имитация времени работы модели (по умолчанию 0.02)

## Демонстрация 429 вручную в Swagger

Очередь разбирается за миллисекунды, поэтому по одному запросу из Swagger она
не переполняется. Чтобы воспроизвести 429 руками, запустите сервис с медленной
моделью и маленькой очередью:

```
QUEUE_MAX_SIZE=2 MODEL_FIXED_OVERHEAD_SECONDS=5 INFERENCE_TIMEOUT_SECONDS=30 python main.py
```

Затем в Swagger быстро (в течение ~5 секунд) отправьте 4-5 запросов
`POST /predict-queued` (удобнее из нескольких вкладок). Первые запросы будут
ждать ответа, а последние сразу получат 429. В это время `GET /queue-status`
показывает `queue_size: 2`.

Автоматический вариант: `python burst_test.py` (400 параллельных запросов).
