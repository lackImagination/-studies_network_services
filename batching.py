# Очередь и dynamic batching

import asyncio
import logging
import time
from dataclasses import dataclass

import config
from model import model_predict_batch
from schemas import FarmRequest

logger = logging.getLogger(__name__)


# Ограниченная очередь: при переполнении endpoint вернет 429 (backpressure)
inference_queue: asyncio.Queue = asyncio.Queue(
    maxsize=config.QUEUE_MAX_SIZE
)


@dataclass
class InferenceTask:
    """Элемент очереди: данные, request_id и Future для ответа."""

    data: FarmRequest
    request_id: str
    future: asyncio.Future


async def batch_worker() -> None:
    """
    Фоновый worker:

    1. Берет первый запрос из очереди.
    2. Копит дополнительные до MAX_BATCH_SIZE
       или до истечения MAX_BATCH_WAIT_MS.
    3. Делает один batch-вызов модели.
    4. Каждому HTTP-запросу возвращает его score через Future.
    """

    logger.info("Batch worker started")

    while True:
        first_task = await inference_queue.get()

        if first_task.future.cancelled():
            inference_queue.task_done()
            continue

        batch = [first_task]

        deadline = time.perf_counter() + config.MAX_BATCH_WAIT_MS / 1000

        # Накопление dynamic batch
        while len(batch) < config.MAX_BATCH_SIZE:
            remaining_time = deadline - time.perf_counter()

            if remaining_time <= 0:
                break

            try:
                task = await asyncio.wait_for(
                    inference_queue.get(),
                    timeout=remaining_time
                )
            except asyncio.TimeoutError:
                break

            if task.future.cancelled():
                inference_queue.task_done()
                continue

            batch.append(task)

        logger.info(
            "Dynamic batch formed | size=%s | queue_size=%s",
            len(batch),
            inference_queue.qsize()
        )

        try:
            scores = await asyncio.wait_for(
                model_predict_batch([task.data for task in batch]),
                timeout=config.INFERENCE_TIMEOUT_SECONDS
            )

            for task, score in zip(batch, scores):
                if not task.future.done():
                    task.future.set_result(score)

        except Exception as exc:
            logger.exception("Batch inference failed")

            for task in batch:
                if not task.future.done():
                    task.future.set_exception(exc)

        finally:
            # На каждый queue.get() должен быть task_done()
            for _ in batch:
                inference_queue.task_done()
