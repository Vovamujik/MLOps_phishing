"""Тесты издателя задач API: очередь tasks, канал с подтверждениями, таймауты, версия брокера."""

import asyncio
import uuid
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest import mock

import aio_pika
import pytest
from faststream.rabbit import RabbitBroker, RabbitMessage, TestRabbitBroker

from mlops_phishing import publisher as publisher_module
from mlops_phishing.config import Settings
from mlops_phishing.messaging import PUBLISH_CHANNEL, TASKS_QUEUE, TaskMessage
from mlops_phishing.publisher import RabbitTaskPublisher

from .conftest import DEAD_RABBITMQ_URL


def make_task() -> TaskMessage:
    return TaskMessage(
        task_id=uuid.uuid4(),
        url="http://ok.example/",
        source="api",
        enqueued_at=datetime.now(UTC),
    )


def healthy_connection(channel: mock.AsyncMock | None = None) -> SimpleNamespace:
    """Открытое соединение: версия в свойствах сервера, канал открывается запросом к брокеру."""
    connected = asyncio.Event()
    connected.set()
    transport = SimpleNamespace(connection=SimpleNamespace(server_properties={"version": "4.3.6"}))
    return SimpleNamespace(
        transport=transport, connected=connected, channel=channel or mock.AsyncMock()
    )


async def hang() -> None:
    await asyncio.sleep(10)


def test_publisher_opens_broker_with_confirming_channel(monkeypatch):
    """Без confirms nack при полной очереди не дошёл бы до API, и 503 не случилось бы."""
    opened: list[dict[str, object]] = []

    def open_broker(url: str, **options: object) -> RabbitBroker:
        opened.append(options)
        return RabbitBroker()

    monkeypatch.setattr(publisher_module, "RabbitBroker", open_broker)
    RabbitTaskPublisher(Settings())

    assert opened == [{"default_channel": PUBLISH_CHANNEL}]


async def test_publish_sends_task_with_its_id_to_tasks_queue():
    publisher = RabbitTaskPublisher(Settings())
    received: list[tuple[TaskMessage, RabbitMessage]] = []

    @publisher.broker.subscriber(TASKS_QUEUE)
    async def handle(task: TaskMessage, message: RabbitMessage) -> None:
        received.append((task, message))

    task = make_task()
    async with TestRabbitBroker(publisher.broker):
        await publisher.publish(task)

    [(delivered, message)] = received
    assert delivered == task
    assert message.message_id == str(task.task_id)


async def test_hanging_broker_does_not_queue_requests(monkeypatch):
    """Брокер не отвечает: каждый запрос ждёт не дольше таймаута, а не по очереди за другими."""
    publisher = RabbitTaskPublisher(Settings(publish_timeout=0.05))
    monkeypatch.setattr(publisher.broker, "connect", hang)
    loop = asyncio.get_running_loop()

    started = loop.time()
    results = await asyncio.gather(
        *(publisher.publish(make_task()) for _ in range(5)), return_exceptions=True
    )
    await publisher.close()

    assert all(isinstance(result, TimeoutError) for result in results)
    # Последовательно вышло бы 5 × 0.05 с
    assert loop.time() - started < 0.2


async def test_concurrent_requests_share_one_connect_attempt(monkeypatch):
    publisher = RabbitTaskPublisher(Settings())
    attempts = 0

    async def slow_connect() -> SimpleNamespace:
        nonlocal attempts
        attempts += 1
        await asyncio.sleep(0.01)
        return healthy_connection()

    monkeypatch.setattr(publisher.broker, "connect", slow_connect)

    connections = await asyncio.gather(*(publisher.connect() for _ in range(5)))

    assert attempts == 1
    assert all(connection is connections[0] for connection in connections)


async def test_failed_connect_is_retried_by_next_request(monkeypatch):
    """Брокер был недоступен — следующий запрос пробует снова, а не получает старую ошибку."""
    publisher = RabbitTaskPublisher(Settings())
    connect = mock.AsyncMock(side_effect=[ConnectionRefusedError("refused"), healthy_connection()])
    monkeypatch.setattr(publisher.broker, "connect", connect)

    with pytest.raises(ConnectionRefusedError):
        await publisher.connect()

    assert await publisher.connect() is not None
    assert connect.await_count == 2


async def test_server_version_round_trips_and_closes_probe_channel(monkeypatch):
    publisher = RabbitTaskPublisher(Settings())
    channel = mock.AsyncMock()
    connection = healthy_connection(channel)
    monkeypatch.setattr(publisher.broker, "connect", mock.AsyncMock(return_value=connection))

    assert await publisher.server_version() == "4.3.6"
    # Канал для проверки закрыт: health не копит каналы на соединении
    channel.return_value.close.assert_awaited_once()


async def test_server_version_during_reconnect_is_connection_error(monkeypatch):
    """Настоящее соединение aio-pika без транспорта, как во время переподключения.

    Его channel() бросает голый RuntimeError, который health превратил бы в 500.
    """
    publisher = RabbitTaskPublisher(Settings())
    reconnecting = aio_pika.RobustConnection(DEAD_RABBITMQ_URL)
    monkeypatch.setattr(publisher.broker, "connect", mock.AsyncMock(return_value=reconnecting))

    with pytest.raises(ConnectionError, match="восстанавливается"):
        await publisher.server_version()


async def test_hanging_broker_is_not_reported_as_healthy(monkeypatch):
    """Соединение живо, но брокер не отвечает на открытие канала: ошибка, а не старая версия."""
    publisher = RabbitTaskPublisher(Settings(publish_timeout=0.05))
    connection = healthy_connection(mock.AsyncMock(side_effect=hang))
    monkeypatch.setattr(publisher.broker, "connect", mock.AsyncMock(return_value=connection))

    with pytest.raises(TimeoutError):
        await publisher.server_version()
