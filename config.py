# Настройки сервиса: batching, очередь, таймауты, SQLite, LLM.

import os
from pathlib import Path


# ------------------------------------------------------------
# BATCHING / QUEUE
# ------------------------------------------------------------

# Максимальное число объектов в одном batch
MAX_BATCH_SIZE = 8

# Максимальное время ожидания накопления dynamic batch
MAX_BATCH_WAIT_MS = 50

# Максимальное количество запросов в очереди (backpressure)
QUEUE_MAX_SIZE = int(os.getenv("QUEUE_MAX_SIZE", "100"))

# Максимальное время ожидания результата инференса
INFERENCE_TIMEOUT_SECONDS = float(
    os.getenv("INFERENCE_TIMEOUT_SECONDS", "2.0")
)


# ------------------------------------------------------------
# ИМИТАЦИЯ ВРЕМЕНИ РАБОТЫ МОДЕЛИ
# ------------------------------------------------------------

# 20 ms - фиксированный overhead одного вызова модели,
# 3 ms  - вычисления на один объект.
MODEL_FIXED_OVERHEAD_SECONDS = float(
    os.getenv("MODEL_FIXED_OVERHEAD_SECONDS", "0.020")
)
MODEL_PER_ITEM_SECONDS = 0.003


# ------------------------------------------------------------
# SQLITE
# ------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DATABASE_PATH = Path(
    os.getenv("DATABASE_PATH", BASE_DIR / "agro_scoring.db")
)

# Если SQLite занята другой транзакцией, соединение ждет до 5 секунд
DB_BUSY_TIMEOUT_MS = 5000


# ------------------------------------------------------------
# LLM (Vireonix, OpenAI-совместимый endpoint без API-ключа)
# ------------------------------------------------------------

LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://vireonix.ai/v1")
LLM_MODEL = "auto"
LLM_TIMEOUT_SECONDS = 60.0
LLM_DEFAULT_MAX_TOKENS = 300
LLM_DEFAULT_TEMPERATURE = 0.3
