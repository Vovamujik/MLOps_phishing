"""Противодавление на настоящем RabbitMQ (Testcontainers): переполнение, отказы, DLQ.

Лимит очереди, overflow=reject-publish, счётчик доставок и DLQ — поведение
самого брокера, TestRabbitBroker его не знает. Топологию задаёт тот же
rabbit_setup, что и rabbitmq-init в compose, только с маленьким лимитом.
"""

import asyncio
import json
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from unittest import mock

import aio_pika
import asyncpg
import pytest
from aio_pika.abc import AbstractChannel, AbstractIncomingMessage
from aio_pika.exceptions import DeliveryError, PublishError
from asgi_lifespan import LifespanManager
from faststream.rabbit import RabbitBroker
from httpx import ASGITransport, AsyncClient

from mlops_phishing import rabbit_setup, tasks_repo
from mlops_phishing.app import app
from mlops_phishing.config import Settings, get_settings
from mlops_phishing.messaging import (
    DEAD_LETTER_QUEUE_NAME,
    PUBLISH_CHANNEL,
    TASKS_QUEUE_NAME,
    TaskMessage,
)
from mlops_phishing.publisher import RabbitTaskPublisher
from mlops_phishing.source import (
    Backoff,
    Row,
    SourceStats,
    broker_publisher,
    run_batch,
    task_id_for,
)
from mlops_phishing.worker import Worker, build_worker

from ..conftest import FakeClassifier, make_model
from .conftest import Rabbit, start_rabbitmq

pytestmark = pytest.mark.integration

LIMIT = 10
DELIVERY_LIMIT = 2
# Сколько ждать событий брокера: доставок, DLQ, возврата сообщений в очередь
WAIT_SECONDS = 15


@pytest.fixture(scope="module")
def rabbit() -> Iterator[Rabbit]:
    with start_rabbitmq() as started:
        yield started


@pytest.fixture(scope="module")
def settings(rabbit: Rabbit) -> Settings:
    """Настройки под контейнер: лимит очереди 10, две повторные доставки до DLQ."""
    settings = Settings(
        _env_file=None,
        rabbitmq_url=rabbit.amqp_url,
        rabbitmq_management_url=rabbit.management_url,
        tasks_queue_max_length=LIMIT,
        tasks_delivery_limit=DELIVERY_LIMIT,
    )
    rabbit_setup.apply_topology(settings)
    return settings


@pytest.fixture
async def channel(settings: Settings) -> AsyncIterator[AbstractChannel]:
    """Канал для проверок; очереди tasks и DLQ пустые перед каждым тестом."""
    connection = await aio_pika.connect(settings.rabbitmq_url)
    channel = await connection.channel()
    for name in (TASKS_QUEUE_NAME, DEAD_LETTER_QUEUE_NAME):
        await (await channel.get_queue(name)).purge()
    yield channel
    await connection.close()


@pytest.fixture
async def publisher(settings: Settings) -> AsyncIterator[RabbitTaskPublisher]:
    publisher = RabbitTaskPublisher(settings)
    yield publisher
    await publisher.close()


@pytest.fixture
async def start_worker(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> AsyncIterator[Callable[..., Awaitable[Worker]]]:
    """Воркер на настоящем брокере; модель и база — заглушки, их тут не проверяем.

    monkeypatch в зависимостях, чтобы воркер остановился раньше, чем вернут
    подменённый в тесте save_result.
    """
    started: list[Worker] = []

    async def start(**overrides: object) -> Worker:
        worker = build_worker(
            settings.model_copy(update={"inference_delay_seconds": 0, **overrides})
        )
        worker.state.model = make_model(FakeClassifier(phishing_probability=0.9))
        worker.state.executor = ThreadPoolExecutor(worker.state.settings.worker_pool_size)
        worker.state.db_pool = mock.Mock()
        # До start: брокер, упавший на полпути, тоже закроется в teardown
        started.append(worker)
        await worker.broker.start()
        return worker

    yield start
    for worker in started:
        await worker.broker.stop()
        if worker.state.executor is not None:
            worker.state.executor.shutdown(wait=True)


def make_task(url: str = "http://paypal-verify.example/login") -> TaskMessage:
    return TaskMessage(task_id=uuid.uuid4(), url=url, source="api", enqueued_at=datetime.now(UTC))


async def depth(channel: AbstractChannel, name: str) -> int:
    """Готовых сообщений в очереди — ровно то, что ограничивает max-length."""
    queue = await channel.declare_queue(name, passive=True)
    return queue.declaration_result.message_count or 0


async def fill_queue(publisher: RabbitTaskPublisher) -> int:
    """Публиковать до первого отказа брокера. Сколько приняли."""
    for accepted in range(LIMIT * 3):
        try:
            await publisher.publish(make_task())
        except DeliveryError:
            return accepted
    raise AssertionError("очередь не ограничена: policy не применилась")


async def next_message(channel: AbstractChannel, name: str) -> AbstractIncomingMessage:
    """Дождаться сообщения в очереди и забрать его."""
    queue = await channel.get_queue(name)
    async with asyncio.timeout(WAIT_SECONDS):
        # Ждём состояние брокера, а не своё событие: остаётся опрос
        while (message := await queue.get(fail=False)) is None:  # noqa: ASYNC110
            await asyncio.sleep(0.05)
    await message.ack()
    return message


async def wait_for_depth(channel: AbstractChannel, name: str, expected: int) -> None:
    async with asyncio.timeout(WAIT_SECONDS):
        while await depth(channel, name) != expected:  # noqa: ASYNC110 — опрос брокера
            await asyncio.sleep(0.05)


async def wait_until(condition: Callable[[], bool]) -> None:
    async with asyncio.timeout(WAIT_SECONDS):
        while not condition():  # noqa: ASYNC110 — счётчики воркера, событий у них нет
            await asyncio.sleep(0.01)


# --- Переполнение: брокер отказывает издателю -------------------------------


async def test_full_queue_nacks_at_once_and_reopens_at_80_percent(publisher, channel):
    """Основа противодавления и Retry-After: отказ сразу, приём снова с 80% лимита."""
    accepted = await fill_queue(publisher)

    # Quorum-очередь включает отказ, когда готовых стало больше лимита
    assert accepted == LIMIT + 1
    assert await depth(channel, TASKS_QUEUE_NAME) == LIMIT + 1
    started = time.perf_counter()
    with pytest.raises(DeliveryError):
        await publisher.publish(make_task())
    assert time.perf_counter() - started < 1

    tasks = await channel.get_queue(TASKS_QUEUE_NAME)
    drained = 0
    while True:
        message = await tasks.get()
        await message.ack()
        drained += 1
        try:
            await publisher.publish(make_task())
            break
        except DeliveryError:
            continue
    assert accepted - drained == LIMIT * 8 // 10


async def test_publish_to_missing_queue_fails_instead_of_losing_task(settings, rabbit):
    """Пустой vhost без очереди: брокер возвращает сообщение, издатель получает ошибку."""
    rabbit.ctl("add_vhost", "empty")
    rabbit.ctl("set_permissions", "-p", "empty", "app", ".*", ".*", ".*")
    publisher = RabbitTaskPublisher(
        settings.model_copy(update={"rabbitmq_url": settings.rabbitmq_url + "empty"})
    )

    with pytest.raises(PublishError):
        await publisher.publish(make_task())
    await publisher.close()


# --- Граница системы: API сбрасывает нагрузку -------------------------------


@pytest.fixture
async def api(settings, postgres_dsn, monkeypatch) -> AsyncIterator[AsyncClient]:
    """Приложение на настоящих RabbitMQ и Postgres."""
    monkeypatch.setenv("RABBITMQ_URL", settings.rabbitmq_url)
    monkeypatch.setenv("DATABASE_URL", postgres_dsn)
    monkeypatch.setenv("TASKS_QUEUE_MAX_LENGTH", str(LIMIT))
    get_settings.cache_clear()
    async with LifespanManager(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            yield client


async def test_api_sheds_load_with_503_and_leaves_no_pending(api, channel, postgres_dsn):
    """Воркера нет: очередь заполняется, дальше 503 с Retry-After, и PENDING только у принятых."""
    marker = uuid.uuid4().hex[:8]
    responses = [
        await api.post("/api/v1/tasks", json={"url": f"http://{marker}-{i}.example/"})
        for i in range(LIMIT + 5)
    ]

    codes = [response.status_code for response in responses]
    assert codes == [202] * (LIMIT + 1) + [503] * 4
    # 20% лимита при 20 задачах/с у одного воркера — меньше секунды, в заголовке минимум 1
    assert {response.headers["Retry-After"] for response in responses[LIMIT + 1 :]} == {"1"}
    connection = await asyncpg.connect(postgres_dsn)
    try:
        pending = await connection.fetch(
            "SELECT task_id FROM tasks WHERE url LIKE $1 AND status = 'PENDING'",
            f"http://{marker}-%",
        )
    finally:
        await connection.close()
    accepted = {uuid.UUID(response.json()["task_id"]) for response in responses[: LIMIT + 1]}
    assert {row["task_id"] for row in pending} == accepted


# --- Источник: противодавление без потерь -----------------------------------


async def test_source_waits_on_full_queue_and_loses_nothing(settings, channel):
    """Медленный потребитель, источник втрое больше лимита: всё доставлено, по порядку, один раз."""
    rows = [Row(number, f"http://batch{number}.example/") for number in range(1, 3 * LIMIT + 1)]
    expected = [str(task_id_for("it", row.number)) for row in rows]
    received: list[str] = []
    depths: list[int] = []
    done = asyncio.Event()

    async def consume(message: AbstractIncomingMessage) -> None:
        await asyncio.sleep(0.02)
        received.append(str(message.message_id))
        await message.ack()
        if len(received) == len(rows):
            done.set()

    async def watch_depth() -> None:
        while not done.is_set():
            depths.append(await depth(channel, TASKS_QUEUE_NAME))
            await asyncio.sleep(0.01)

    consumer_connection = await aio_pika.connect(settings.rabbitmq_url)
    consumer_channel = await consumer_connection.channel()
    await consumer_channel.set_qos(prefetch_count=1)
    await (await consumer_channel.get_queue(TASKS_QUEUE_NAME)).consume(consume)
    watcher = asyncio.create_task(watch_depth())
    stats = SourceStats()
    try:
        async with RabbitBroker(settings.rabbitmq_url, default_channel=PUBLISH_CHANNEL) as broker:
            batch = run_batch(
                rows, broker_publisher(broker), "it", Backoff(0.01, 0.05), asyncio.Event(), stats
            )
            await asyncio.wait_for(batch, timeout=30)
        await asyncio.wait_for(done.wait(), timeout=30)
        await watcher
    finally:
        watcher.cancel()
        await consumer_connection.close()

    assert received == expected
    assert stats.nacked > 0, "источник ни разу не упёрся в лимит"
    assert max(depths) <= LIMIT + 1


# --- Воркер: повторы, DLQ, prefetch, остановка ------------------------------


async def test_failing_task_goes_to_dlq_after_delivery_limit(
    start_worker, publisher, channel, monkeypatch
):
    """База недоступна при каждой попытке: 1 + delivery-limit доставок, потом DLQ."""
    attempts: list[uuid.UUID] = []

    async def database_down(pool: object, task_id: uuid.UUID, *args: object) -> bool:
        attempts.append(task_id)
        raise OSError("connection refused")

    monkeypatch.setattr(tasks_repo, "save_result", database_down)
    await start_worker()
    task = make_task()

    await publisher.publish(task)
    dead = await next_message(channel, DEAD_LETTER_QUEUE_NAME)

    assert json.loads(dead.body)["task_id"] == str(task.task_id)
    assert attempts == [task.task_id] * (DELIVERY_LIMIT + 1)
    # Сам брокер считает, сколько раз отдал сообщение, и пишет, почему отправил в DLQ
    assert dead.headers["x-acquired-count"] == DELIVERY_LIMIT + 1
    assert dead.headers["x-first-death-reason"] == "delivery_limit"
    assert await depth(channel, TASKS_QUEUE_NAME) == 0


async def test_malformed_message_goes_to_dlq_without_retries(start_worker, channel, monkeypatch):
    save = mock.AsyncMock(return_value=True)
    monkeypatch.setattr(tasks_repo, "save_result", save)
    await start_worker()

    await channel.default_exchange.publish(
        aio_pika.Message(b'{"garbage": 1}', content_type="application/json"),
        routing_key=TASKS_QUEUE_NAME,
    )
    dead = await next_message(channel, DEAD_LETTER_QUEUE_NAME)

    assert json.loads(dead.body) == {"garbage": 1}
    # Одна доставка и reject без возврата: повтор битое сообщение не исправит
    assert dead.headers["x-acquired-count"] == 1
    assert dead.headers["x-first-death-reason"] == "rejected"
    save.assert_not_awaited()


async def test_prefetch_bounds_tasks_in_work(start_worker, publisher, channel, monkeypatch):
    """Потоков 4, prefetch 2: брокер не отдаёт воркеру больше двух задач сразу."""
    monkeypatch.setattr(tasks_repo, "save_result", mock.AsyncMock(return_value=True))
    worker = await start_worker(worker_pool_size=4, worker_prefetch=2, inference_delay_seconds=0.1)
    peak = 0

    for _ in range(8):
        await publisher.publish(make_task())

    def all_processed() -> bool:
        nonlocal peak
        peak = max(peak, worker.state.stats.in_flight)
        return worker.state.stats.processed == 8

    await wait_until(all_processed)
    assert peak == 2


async def test_graceful_stop_finishes_tasks_in_work(start_worker, publisher, channel, monkeypatch):
    """Остановка как по SIGTERM: взятые в работу дописаны, остальные ждут в очереди."""
    save = mock.AsyncMock(return_value=True)
    monkeypatch.setattr(tasks_repo, "save_result", save)
    worker = await start_worker(worker_pool_size=4, worker_prefetch=4, inference_delay_seconds=0.5)
    for _ in range(8):
        await publisher.publish(make_task())
    await wait_until(lambda: worker.state.stats.in_flight == 4)

    await worker.broker.stop()

    assert save.await_count == 4
    assert worker.state.stats.in_flight == 0
    # Пока дописывались взятые, брокер дослал воркеру задачи по prefetch. Воркер их не брал,
    # закрытие канала вернуло их в очередь: ничего не потеряно и не обработано дважды
    await wait_for_depth(channel, TASKS_QUEUE_NAME, 4)
    returned = [await next_message(channel, TASKS_QUEUE_NAME) for _ in range(4)]
    assert [message.headers["x-acquired-count"] for message in returned] == [1] * 4
    assert await depth(channel, DEAD_LETTER_QUEUE_NAME) == 0
