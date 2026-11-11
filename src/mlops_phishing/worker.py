"""Воркер конвейера: inference модели из MLflow над задачами из очереди.

Противодавление «брокер → воркер»: prefetch ограничивает, сколько
неподтверждённых задач брокер отдаёт воркеру. FastStream запускает обработчик
на каждую доставку, поэтому одновременно в работе ровно prefetch задач, и
prefetch равен числу потоков пула: каждой взятой задаче сразу есть поток.

Подтверждение — после записи результата в базу (AckPolicy.REJECT_ON_ERROR):
- успех: ack после записи; упади воркер раньше, задача вернётся в очередь;
- база или сеть недоступны: reject(requeue=True). Это считается попыткой,
  и после delivery-limit задача уходит в DLQ. nack(requeue=True) в RabbitMQ 4.3
  попыткой не считается и вернул бы задачу в очередь бесконечно;
- битое сообщение или ошибка в коде: reject без возврата, сразу в DLQ —
  повтор дал бы ту же ошибку.

Запуск: faststream run mlops_phishing.worker:app --host 0.0.0.0 --port 8001
"""

import asyncio
import json
import logging
import time
from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass, field

import asyncpg
from faststream import AckPolicy
from faststream.asgi import AsgiFastStream, AsgiResponse, get, make_ping_asgi
from faststream.exceptions import RejectMessage
from faststream.rabbit import RabbitBroker
from faststream.rabbit.schemas import Channel

from mlops_phishing import logging_config, ml, tasks_repo
from mlops_phishing.config import Settings, get_settings
from mlops_phishing.messaging import TASKS_QUEUE, TaskMessage
from mlops_phishing.tasks_repo import TaskResult

log = logging.getLogger(__name__)

# Временные отказы базы или сети: задачу стоит повторить
RETRYABLE_ERRORS = (asyncpg.PostgresError, OSError, TimeoutError)


@dataclass
class WorkerStats:
    """Счётчики для /stats: видно, что в работе ровно prefetch задач."""

    in_flight: int = 0
    processed: int = 0
    duplicates: int = 0
    retried: int = 0


@dataclass
class WorkerState:
    settings: Settings
    stats: WorkerStats = field(default_factory=WorkerStats)
    model: ml.LoadedModel | None = None
    executor: ThreadPoolExecutor | None = None
    db_pool: asyncpg.Pool | None = None


@dataclass(frozen=True)
class Worker:
    broker: RabbitBroker
    app: AsgiFastStream
    state: WorkerState


def score(model: ml.LoadedModel, url: str, delay: float) -> float:
    """Inference в потоке пула. Задержка имитирует тяжёлую модель, задаётся конфигом.

    Вызывается только через run_in_executor: sleep здесь держит поток пула, а
    не event loop, поэтому воркер в это время отвечает на /health.
    """
    probability = model.phishing_probability(url)
    if delay > 0:
        time.sleep(delay)
    return probability


async def process(state: WorkerState, task: TaskMessage) -> None:
    """Посчитать задачу и записать результат. Ack — после записи."""
    if state.model is None or state.executor is None or state.db_pool is None:
        raise RuntimeError("Воркер не инициализирован: lifespan не выполнился")
    stats = state.stats
    stats.in_flight += 1
    try:
        loop = asyncio.get_running_loop()
        probability = await loop.run_in_executor(
            state.executor,
            score,
            state.model,
            task.url,
            state.settings.inference_delay_seconds,
        )
        result = TaskResult(
            is_phishing=probability >= state.settings.phishing_threshold,
            phishing_probability=probability,
            model_version=state.model.version,
        )
        try:
            written = await tasks_repo.save_result(
                state.db_pool, task.task_id, task.url, task.source, task.enqueued_at, result
            )
        except RETRYABLE_ERRORS as exc:
            stats.retried += 1
            log.warning(
                "Задача %s: результат не записан (%s), вернём в очередь",
                task.task_id,
                type(exc).__name__,
            )
            raise RejectMessage(requeue=True) from exc

        if written:
            stats.processed += 1
        else:
            # Повторная доставка уже готовой задачи: результат не трогаем, просто ack
            stats.duplicates += 1
            log.info("Задача %s уже готова, повторная доставка", task.task_id)
    finally:
        stats.in_flight -= 1


def build_worker(settings: Settings) -> Worker:
    state = WorkerState(settings=settings)
    broker = RabbitBroker(settings.rabbitmq_url, graceful_timeout=settings.worker_graceful_timeout)

    @broker.subscriber(
        TASKS_QUEUE,
        # prefetch 0 в настройках значит «без ограничения»: basic.qos не отправляется
        channel=Channel(prefetch_count=settings.worker_prefetch or None),
        ack_policy=AckPolicy.REJECT_ON_ERROR,
    )
    async def handle(task: TaskMessage) -> None:
        await process(state, task)

    @asynccontextmanager
    async def lifespan() -> AsyncIterator[None]:
        """До первого сообщения: модель по alias, пул потоков, пул БД.

        Без модели воркер не стартует: брать задачи, которые нечем считать,
        бессмысленно. Закрывается после того, как брокер дождался задач в работе.
        """
        logging_config.setup_logging(settings.log_level)
        state.model = await asyncio.to_thread(ml.load_model, settings)
        state.executor = ThreadPoolExecutor(
            settings.worker_pool_size, thread_name_prefix="inference"
        )
        state.db_pool = await asyncpg.create_pool(
            settings.database_url, min_size=1, max_size=settings.worker_pool_size
        )
        await tasks_repo.ensure_schema(state.db_pool)
        log.info(
            "Воркер готов: модель %s v%s (@%s), пул %d, prefetch %s, задержка %.2f с",
            state.model.name,
            state.model.version,
            state.model.alias,
            settings.worker_pool_size,
            settings.worker_prefetch or "без ограничения",
            settings.inference_delay_seconds,
        )
        try:
            yield
        finally:
            state.executor.shutdown(wait=True)
            await state.db_pool.close()
            log.info("Воркер остановлен: %s", asdict(state.stats))

    @get
    async def stats(scope: object) -> AsgiResponse:
        body = asdict(state.stats) | {
            "prefetch": settings.worker_prefetch,
            "pool_size": settings.worker_pool_size,
            "model_version": state.model.version if state.model else None,
        }
        return AsgiResponse(json.dumps(body).encode(), status_code=200)

    app = AsgiFastStream(
        broker,
        asgi_routes=[("/health", make_ping_asgi(broker, timeout=2.0)), ("/stats", stats)],
        lifespan=lifespan,
    )
    return Worker(broker=broker, app=app, state=state)


worker = build_worker(get_settings())
app = worker.app
