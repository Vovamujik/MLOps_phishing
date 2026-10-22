"""Тесты POST и GET /api/v1/tasks: 202 и Location, 503 с Retry-After, нет PENDING после отказа.

Брокер изображает FakePublisher: отказ при полной очереди — это nack, который
TestRabbitBroker не умеет. На настоящем RabbitMQ — в интеграционных тестах.
"""

import logging
import time
import uuid
from datetime import UTC, datetime
from unittest import mock

import pytest
from aio_pika.exceptions import ChannelInvalidStateError

from mlops_phishing import tasks_repo
from mlops_phishing.api.v1.tasks import retry_after_seconds
from mlops_phishing.app import app
from mlops_phishing.config import Settings
from mlops_phishing.tasks_repo import TaskRecord, TaskStatus

from .conftest import FakePublisher, broker_nack, broker_return

URL = "http://paypal-secure-login.verify-account.xyz/signin"


@pytest.fixture
def pending(monkeypatch: pytest.MonkeyPatch) -> mock.AsyncMock:
    """Запись строки PENDING: запоминает вызовы вместо базы."""
    insert = mock.AsyncMock()
    monkeypatch.setattr(tasks_repo, "insert_pending", insert)
    return insert


# --- POST /api/v1/tasks ---------------------------------------------------


async def test_accepted_task_gets_202_location_and_pending_row(client, install_publisher, pending):
    publisher = install_publisher(FakePublisher())

    response = await client.post("/api/v1/tasks", json={"url": URL})
    body = response.json()

    assert response.status_code == 202
    assert body["status"] == "PENDING"
    assert response.headers["Location"] == body["status_url"]
    assert body["status_url"].endswith(f"/api/v1/tasks/{body['task_id']}")
    [task] = publisher.tasks
    assert (str(task.task_id), task.url, task.source) == (body["task_id"], URL, "api")
    _, task_id, url, enqueued_at = pending.await_args.args
    assert (task_id, url, enqueued_at) == (task.task_id, URL, task.enqueued_at)


async def test_full_queue_answers_503_with_retry_after_and_leaves_no_pending(
    client, install_publisher, pending
):
    """Клиента замедлить нельзя: отказ сразу, без повторов и без записи в базу."""
    publisher = install_publisher(FakePublisher(error=broker_nack()))

    started = time.perf_counter()
    response = await client.post("/api/v1/tasks", json={"url": URL})

    assert response.status_code == 503
    assert response.headers["Retry-After"] == "1"
    assert "заполнена" in response.json()["detail"]
    assert len(publisher.tasks) == 1
    pending.assert_not_awaited()
    assert time.perf_counter() - started < 0.5


@pytest.mark.parametrize(
    "error",
    [ConnectionRefusedError("refused"), TimeoutError(), ChannelInvalidStateError("closed")],
    ids=["refused", "timeout", "channel-closed"],
)
async def test_unavailable_broker_answers_503_without_retry_after(
    client, install_publisher, pending, error
):
    """Когда брокер поднимется, не знаем — Retry-After только для переполнения."""
    install_publisher(FakePublisher(error=error))

    response = await client.post("/api/v1/tasks", json={"url": URL})

    assert response.status_code == 503
    assert "Retry-After" not in response.headers
    pending.assert_not_awaited()


async def test_missing_queue_answers_503(client, install_publisher, pending):
    install_publisher(FakePublisher(error=broker_return()))

    response = await client.post("/api/v1/tasks", json={"url": URL})

    assert response.status_code == 503
    assert response.json()["detail"] == "Очередь задач не настроена"
    pending.assert_not_awaited()


async def test_real_publisher_without_broker_answers_503_quickly(client, pending):
    """Настоящий издатель, RabbitMQ на закрытом порту: 503 сразу, запрос не висит."""
    started = time.perf_counter()
    response = await client.post("/api/v1/tasks", json={"url": URL})

    assert response.status_code == 503
    assert time.perf_counter() - started < Settings().publish_timeout
    pending.assert_not_awaited()


async def test_database_failure_after_publish_still_accepts(
    client, install_publisher, pending, caplog
):
    """Задача уже в очереди: воркер создаст строку сам, поэтому 202, а не 503."""
    caplog.set_level(logging.WARNING, logger="mlops_phishing.api.v1.tasks")
    install_publisher(FakePublisher())
    pending.side_effect = OSError("connection refused")

    response = await client.post("/api/v1/tasks", json={"url": URL})

    assert response.status_code == 202
    assert "строка PENDING не записана" in caplog.text


@pytest.mark.parametrize(
    "url",
    [
        "",
        "   ",
        "paypal.com/login",
        "ftp://files.example.com",
        "https://",
        "https://" + "a" * 2050,
        "http://a.example/\x00",  # NUL база не примет уже после публикации
    ],
)
async def test_invalid_url_is_rejected_before_queue(client, install_publisher, url):
    publisher = install_publisher(FakePublisher())

    response = await client.post("/api/v1/tasks", json={"url": url})

    assert response.status_code == 422
    assert publisher.tasks == []


async def test_whitespace_around_url_is_trimmed(client, install_publisher, pending):
    publisher = install_publisher(FakePublisher())

    await client.post("/api/v1/tasks", json={"url": f"  {URL}\n"})

    assert publisher.tasks[0].url == URL


async def test_url_is_not_logged(client, install_publisher, pending, caplog):
    """URL — данные пользователя: в логе только id задачи и коды ответов."""
    caplog.set_level(logging.DEBUG, logger="mlops_phishing")
    install_publisher(FakePublisher(error=broker_nack()))

    await client.post("/api/v1/tasks", json={"url": URL})

    assert "Retry-After" in caplog.text
    assert URL not in caplog.text


@pytest.mark.parametrize(
    ("max_length", "delay", "pool_size", "prefetch", "expected"),
    [
        (100, 0.2, 4, 4, 1),  # по умолчанию: 20 задач при 20 задачах/с
        (1000, 0.2, 4, 4, 10),
        (100, 0.2, 1, 1, 4),
        (100, 0.2, 4, 1, 4),  # prefetch 1: воркер считает по одной задаче, 5 задач/с
        (100, 0.2, 4, 0, 1),  # prefetch 0 — без лимита, упор в пул
        (100, 0.25, 4, 4, 2),  # 1.25 с — вверх, иначе клиент придёт раньше
        (100, 0.0, 4, 4, 1),  # без задержки — минимум 1 с, не 0
    ],
)
def test_retry_after_is_time_to_drain_fifth_of_limit(
    max_length, delay, pool_size, prefetch, expected
):
    settings = Settings(
        tasks_queue_max_length=max_length,
        inference_delay_seconds=delay,
        worker_pool_size=pool_size,
        worker_prefetch=prefetch,
    )

    assert retry_after_seconds(settings) == expected


async def test_retry_after_header_follows_settings(client, install_publisher, monkeypatch):
    """Заголовок считается из настроек, а не захардкожен: лимит 1000 — 10 с."""
    monkeypatch.setattr(app.state, "settings", Settings(tasks_queue_max_length=1000))
    install_publisher(FakePublisher(error=broker_nack()))

    response = await client.post("/api/v1/tasks", json={"url": URL})

    assert response.headers["Retry-After"] == "10"


# --- GET /api/v1/tasks/{task_id} ------------------------------------------


def make_record(status: TaskStatus) -> TaskRecord:
    done = status is TaskStatus.DONE
    return TaskRecord(
        task_id=uuid.uuid4(),
        url=URL,
        source="api",
        status=status,
        is_phishing=True if done else None,
        phishing_probability=0.97 if done else None,
        model_version="2" if done else None,
        enqueued_at=datetime(2026, 10, 20, 12, 0, tzinfo=UTC),
        completed_at=datetime(2026, 10, 20, 12, 0, 1, tzinfo=UTC) if done else None,
    )


async def test_done_task_shows_result(client, monkeypatch):
    record = make_record(TaskStatus.DONE)
    monkeypatch.setattr(tasks_repo, "get_task", mock.AsyncMock(return_value=record))

    response = await client.get(f"/api/v1/tasks/{record.task_id}")
    body = response.json()

    assert response.status_code == 200
    assert body["status"] == "DONE"
    assert (body["is_phishing"], body["phishing_probability"], body["model_version"]) == (
        True,
        0.97,
        "2",
    )
    assert body["completed_at"] == "2026-10-20T12:00:01Z"


async def test_pending_task_has_no_result_yet(client, monkeypatch):
    record = make_record(TaskStatus.PENDING)
    monkeypatch.setattr(tasks_repo, "get_task", mock.AsyncMock(return_value=record))

    body = (await client.get(f"/api/v1/tasks/{record.task_id}")).json()

    assert body["status"] == "PENDING"
    assert body["is_phishing"] is None
    assert body["completed_at"] is None


async def test_unknown_task_is_404(client, monkeypatch):
    monkeypatch.setattr(tasks_repo, "get_task", mock.AsyncMock(return_value=None))

    response = await client.get(f"/api/v1/tasks/{uuid.uuid4()}")

    assert response.status_code == 404


async def test_malformed_task_id_is_422(client):
    assert (await client.get("/api/v1/tasks/not-a-uuid")).status_code == 422


async def test_database_down_on_read_is_503(client):
    """База в тестах на закрытом порту: честный 503, а не 500."""
    response = await client.get(f"/api/v1/tasks/{uuid.uuid4()}")

    assert response.status_code == 503


async def test_load_shedding_access_line_is_a_warning(client, install_publisher, caplog):
    """Под перегрузкой лог не должен выглядеть как авария: 503 с Retry-After — WARNING."""
    caplog.set_level(logging.DEBUG, logger="mlops_phishing.app")
    install_publisher(FakePublisher(error=broker_nack()))

    await client.post("/api/v1/tasks", json={"url": URL})

    [access] = [
        record
        for record in caplog.records
        if record.name == "mlops_phishing.app" and "POST /api/v1/tasks" in record.getMessage()
    ]
    assert access.levelno == logging.WARNING
