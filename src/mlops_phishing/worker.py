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

Контекст задачи: middleware на время сообщения кладёт в contextvars task_id (из
correlation_id) и номер попытки (из x-delivery-count quorum-очереди), поэтому они есть
во всех записях обработчика без передачи руками. Он же логирует перевод в DLQ.

Запуск: faststream run mlops_phishing.worker:app --host 0.0.0.0 --port 8001
"""

import asyncio
import json
import logging
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass, field
from types import TracebackType
from typing import Any

import aio_pika
import asyncpg
import structlog
from faststream import AckPolicy, BaseMiddleware, ContextRepo
from faststream.asgi import AsgiFastStream, AsgiResponse, get, make_ping_asgi
from faststream.exceptions import RejectMessage
from faststream.message import StreamMessage
from faststream.rabbit import RabbitBroker
from faststream.rabbit.schemas import Channel

from mlops_phishing import logging_config, ml, tasks_repo
from mlops_phishing.config import Settings, get_settings
from mlops_phishing.messaging import TASKS_QUEUE, TaskMessage
from mlops_phishing.tasks_repo import TaskResult

log = logging.getLogger(__name__)

# Временные отказы базы или сети: задачу стоит повторить
RETRYABLE_ERRORS = (asyncpg.PostgresError, OSError, TimeoutError)

# Заголовок quorum-очереди: сколько раз задачу уже вернули (reject с возвратом)
DELIVERY_COUNT_HEADER = "x-delivery-count"

RawMessage = aio_pika.IncomingMessage
ConsumeNext = Callable[[StreamMessage[RawMessage]], Awaitable[object]]


@dataclass
class WorkerStats:
    """Счётчики для /stats: видно, что в работе ровно prefetch задач."""

    in_flight: int = 0
    processed: int = 0
    duplicates: int = 0
    retried: int = 0
    dead_lettered: int = 0


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


def delivery_count(headers: dict[str, Any] | None) -> int:
    """Сколько раз задачу уже возвращали в очередь: 0 у первой доставки."""
    value = (headers or {}).get(DELIVERY_COUNT_HEADER, 0)
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


class TaskContextMiddleware(BaseMiddleware):
    """task_id и номер попытки — в contextvars на время сообщения; итог — в лог.

    Брокер создаёт экземпляр на каждое сообщение. task_id берётся из correlation_id
    (издатели ставят его равным task_id), у сообщений без него — из message_id.
    """

    def __init__(
        self, msg: RawMessage | None, /, *, context: ContextRepo, state: WorkerState
    ) -> None:
        super().__init__(msg, context=context)
        self.state = state
        self.redeliveries = delivery_count(getattr(msg, "headers", None))
        self.attempt = self.redeliveries + 1
        self._tokens: dict[str, Any] = {}

    async def on_receive(self) -> None:
        task_id = getattr(self.msg, "correlation_id", None) or getattr(self.msg, "message_id", None)
        fields: dict[str, object] = {"attempt": self.attempt}
        if task_id:
            fields["task_id"] = str(task_id)
        self._tokens = structlog.contextvars.bind_contextvars(**fields)
        log.debug("Задача получена")
        if self.redeliveries:
            log.info("Повторная доставка задачи")

    async def consume_scope(  # type: ignore[override]
        self, call_next: ConsumeNext, msg: StreamMessage[RawMessage]
    ) -> object:
        """Перевод в DLQ — ERROR: задача больше не повторится, нужен человек."""
        try:
            return await call_next(msg)
        except RejectMessage as exc:
            if not exc.extra_options.get("requeue"):
                self.dead_letter("отклонена без повтора")
            elif self.redeliveries >= self.state.settings.tasks_delivery_limit:
                # Это возврат сверх delivery-limit: брокер переложит задачу в DLQ
                self.dead_letter("исчерпаны попытки")
            raise
        except Exception:
            # Битое сообщение или ошибка в коде: reject без возврата, сразу в DLQ
            self.state.stats.dead_lettered += 1
            log.exception("Ошибка обработки, задача переведена в DLQ без повторов")
            raise

    def dead_letter(self, reason: str) -> None:
        self.state.stats.dead_lettered += 1
        log.error(
            "Задача переведена в DLQ: %s",
            reason,
            extra={"delivery_limit": self.state.settings.tasks_delivery_limit},
        )

    async def after_processed(
        self,
        exc_type: type[BaseException] | None = None,
        exc_val: BaseException | None = None,
        exc_tb: TracebackType | None = None,
    ) -> bool | None:
        structlog.contextvars.reset_contextvars(**self._tokens)
        return False


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
            # Номер попытки и task_id — из contextvars (TaskContextMiddleware)
            log.warning("Результат не записан, задача вернётся в очередь", exc_info=exc)
            raise RejectMessage(requeue=True) from exc

        if written:
            stats.processed += 1
            log.info(
                "Задача обработана",
                extra={"model_version": result.model_version, "is_phishing": result.is_phishing},
            )
        else:
            # Повторная доставка уже готовой задачи: результат не трогаем, просто ack
            stats.duplicates += 1
            log.info("Задача уже готова, повторная доставка: результат не меняем")
    finally:
        stats.in_flight -= 1


def build_worker(settings: Settings) -> Worker:
    state = WorkerState(settings=settings)

    def task_context(msg: RawMessage | None, /, *, context: ContextRepo) -> TaskContextMiddleware:
        return TaskContextMiddleware(msg, context=context, state=state)

    broker = RabbitBroker(
        settings.rabbitmq_url,
        graceful_timeout=settings.worker_graceful_timeout,
        middlewares=[task_context],
        # Свои строки FastStream о каждом сообщении («Received», «Processed», ошибка со
        # стектрейсом) дублировали бы TaskContextMiddleware, где есть task_id и попытка
        logger=None,
    )

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
        state.model = await asyncio.to_thread(ml.load_model, settings)
        log.info(
            "Модель загружена: %s v%s (@%s)",
            state.model.name,
            state.model.version,
            state.model.alias,
            extra={
                "model_name": state.model.name,
                "model_alias": state.model.alias,
                "model_version": state.model.version,
                "model_run_id": state.model.run_id,
            },
        )
        state.executor = ThreadPoolExecutor(
            settings.worker_pool_size, thread_name_prefix="inference"
        )
        state.db_pool = await asyncpg.create_pool(
            settings.database_url, min_size=1, max_size=settings.worker_pool_size
        )
        await tasks_repo.ensure_schema(state.db_pool)
        log.info(
            "Воркер готов: пул %d, prefetch %s, задержка %.2f с",
            settings.worker_pool_size,
            settings.worker_prefetch or "без ограничения",
            settings.inference_delay_seconds,
        )
        try:
            yield
        finally:
            state.executor.shutdown(wait=True)
            await state.db_pool.close()
            log.info("Ресурсы воркера закрыты", extra=asdict(state.stats))

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

    @app.on_shutdown
    def shutdown_started() -> None:
        # SIGTERM: брокер перестаёт брать задачи и ждёт взятые не дольше graceful_timeout
        log.info(
            "Остановка воркера: новые задачи не берём, ждём %d в работе (до %.0f с)",
            state.stats.in_flight,
            settings.worker_graceful_timeout,
            extra={"in_flight": state.stats.in_flight},
        )

    @app.after_shutdown
    def shutdown_finished() -> None:
        log.info(
            "Воркер остановлен: в работе осталось %d",
            state.stats.in_flight,
            extra=asdict(state.stats),
        )

    return Worker(broker=broker, app=app, state=state)


_settings = get_settings()
# При импорте: `faststream run` пишет о старте раньше lifespan
logging_config.setup_logging(_settings.log_level, "worker", _settings.environment)
worker = build_worker(_settings)
app = worker.app
