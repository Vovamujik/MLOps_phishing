"""Метрики: счётчики и гистограммы API, источника и воркера, границы бакетов, имена.

Провайдер с InMemoryMetricReader ставится в tests/__init__; значения кумулятивные
за весь прогон, поэтому тесты сравнивают «до» и «после».
"""

import asyncio
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from unittest import mock

import pytest
from faststream.rabbit import TestRabbitBroker
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from pydantic import ValidationError

from mlops_phishing import messaging, source, tasks_repo, telemetry
from mlops_phishing.config import Settings
from mlops_phishing.worker import build_worker

from .conftest import (
    FakePublisher,
    broker_nack,
    histogram_count,
    make_model,
    metric_points,
    metric_value,
)
from .test_source import FakeBroker, make_rows, publish_rows


@pytest.fixture
def worker():
    built = build_worker(Settings(inference_delay_seconds=0, worker_pool_size=2))
    built.state.model = make_model()
    built.state.executor = ThreadPoolExecutor(2)
    built.state.db_pool = mock.Mock()
    yield built
    built.state.executor.shutdown(wait=True)


def make_task(**overrides: object) -> messaging.TaskMessage:
    fields: dict[str, object] = {
        "task_id": uuid.uuid4(),
        "url": "http://a.example/",
        "source": "batch",
        "enqueued_at": datetime.now(UTC),
    }
    return messaging.TaskMessage.model_validate(fields | overrides)


async def deliver(worker, task: messaging.TaskMessage, delivery_count: int = 0) -> None:
    async with TestRabbitBroker(worker.broker) as broker:
        await broker.publish(
            task, queue=messaging.TASKS_QUEUE, headers={"x-delivery-count": delivery_count}
        )


# --- Воркер -------------------------------------------------------------------


async def test_processed_task_counts_ok_and_records_durations(worker, monkeypatch):
    monkeypatch.setattr(tasks_repo, "save_result", mock.AsyncMock(return_value=True))
    ok = metric_value("worker.tasks.processed", result="ok")
    inference = histogram_count("worker.inference.duration")
    task_duration = histogram_count("worker.task.duration")
    in_flight = metric_value("worker.tasks.in_flight")

    # Задача ждала в очереди 3 с: это попадёт в worker.task.duration
    await deliver(worker, make_task(enqueued_at=datetime.now(UTC) - timedelta(seconds=3)))

    assert metric_value("worker.tasks.processed", result="ok") == ok + 1
    assert histogram_count("worker.inference.duration") == inference + 1
    assert histogram_count("worker.task.duration") == task_duration + 1
    [point] = metric_points("worker.task.duration")
    assert point.max >= 3
    assert metric_value("worker.tasks.in_flight") == in_flight


async def test_retry_counts_error_and_last_attempt_counts_dlq(worker, monkeypatch):
    monkeypatch.setattr(tasks_repo, "save_result", mock.AsyncMock(side_effect=OSError("down")))
    limit = worker.state.settings.tasks_delivery_limit
    error = metric_value("worker.tasks.processed", result="error")
    dlq = metric_value("worker.tasks.processed", result="dlq")
    redeliveries = metric_value("worker.redeliveries")

    await deliver(worker, make_task())
    await deliver(worker, make_task(), delivery_count=1)
    await deliver(worker, make_task(), delivery_count=limit)

    assert metric_value("worker.tasks.processed", result="error") == error + 2
    assert metric_value("worker.tasks.processed", result="dlq") == dlq + 1
    assert metric_value("worker.redeliveries") == redeliveries + 2


async def test_malformed_message_counts_dlq(worker):
    dlq = metric_value("worker.tasks.processed", result="dlq")

    async with TestRabbitBroker(worker.broker) as broker:
        with pytest.raises(ValidationError):
            await broker.publish({"url": "no task id"}, queue=messaging.TASKS_QUEUE)

    assert metric_value("worker.tasks.processed", result="dlq") == dlq + 1


def test_result_is_the_only_worker_label():
    """Кардинальность: у processed ровно три ряда на воркер, без task_id."""
    for point in metric_points("worker.tasks.processed"):
        assert set(point.attributes) == {"result"}
        assert point.attributes["result"] in {"ok", "error", "dlq"}


# --- Источник ---------------------------------------------------------------


async def test_source_counts_published_rejections_and_wait():
    published = metric_value("source.tasks.published")
    rejections = metric_value("source.broker.rejections")
    waited = metric_value("source.backoff.duration")

    stats = await publish_rows(make_rows(2), FakeBroker(full=3).publish)

    assert metric_value("source.tasks.published") == published + 2
    assert metric_value("source.broker.rejections") == rejections + 3
    assert metric_value("source.backoff.duration") == pytest.approx(waited + stats.waited)


# --- API ----------------------------------------------------------------------


async def test_api_request_duration_by_route_and_status(client, install_publisher):
    install_publisher(FakePublisher(error=broker_nack()))
    attributes = {"http.route": "/api/v1/tasks", "http.response.status_code": 503}
    before = histogram_count("http.server.request.duration", **attributes)

    await client.post("/api/v1/tasks", json={"url": "http://a.example/"})
    await client.get("/healthz")

    assert histogram_count("http.server.request.duration", **attributes) == before + 1
    [point] = metric_points("http.server.request.duration", **attributes)
    assert point.attributes["http.request.method"] == "POST"
    assert tuple(point.explicit_bounds) == telemetry.HTTP_DURATION_BUCKETS
    # Пробы не пишут ни трейсов, ни метрик
    assert not metric_points("http.server.request.duration", **{"http.route": "/healthz"})


async def test_queue_limit_gauge_follows_settings(client):
    [point] = metric_points("tasks.queue.limit")

    assert point.value == Settings().tasks_queue_max_length


def test_duplicate_library_metrics_are_dropped():
    assert not metric_points("messaging.process.duration")
    assert not metric_points("messaging.publish.duration")


# --- Бакеты и имена ---------------------------------------------------------


def test_inference_buckets_are_dense_around_configured_delay():
    buckets = telemetry.inference_buckets(0.2)

    assert list(buckets) == sorted(buckets)
    near = [b for b in buckets if 0.2 <= b <= 0.3]
    assert len(near) >= 6
    assert buckets[0] < 0.1
    assert buckets[-1] == 2.0
    assert telemetry.inference_buckets(0) == telemetry.INFERENCE_FALLBACK_BUCKETS


async def test_inference_histogram_uses_configured_buckets(worker, monkeypatch):
    monkeypatch.setattr(tasks_repo, "save_result", mock.AsyncMock(return_value=True))

    await deliver(worker, make_task())

    [point] = metric_points("worker.inference.duration")
    # Провайдер тестов собран с задержкой по умолчанию — 0.2 с
    assert tuple(point.explicit_bounds) == telemetry.inference_buckets(0.2)


def test_process_metrics_are_memory_and_cpu_only():
    """Ровно те имена, что уйдут в Prometheus как process_memory_usage_bytes и т. д."""
    reader = InMemoryMetricReader()
    provider = MeterProvider(metric_readers=[reader])
    instrumentor = telemetry.SystemMetricsInstrumentor(config=telemetry.PROCESS_METRICS)
    instrumentor.instrument(meter_provider=provider)
    try:
        data = reader.get_metrics_data()
    finally:
        instrumentor.uninstrument()

    metrics_by_name = {
        metric.name: metric
        for resource in data.resource_metrics
        for scope in resource.scope_metrics
        for metric in scope.metrics
    }
    assert set(metrics_by_name) == {"process.memory.usage", "process.cpu.time"}
    assert metrics_by_name["process.memory.usage"].unit == "By"
    [memory] = metrics_by_name["process.memory.usage"].data.data_points
    assert memory.value > 0


async def test_short_batch_still_reports_metrics_on_exit(monkeypatch):
    """Источник сбрасывает метрики при выходе: короткая партия не теряется до экспорта."""
    shutdown = mock.Mock()
    monkeypatch.setattr(telemetry, "setup_telemetry", lambda service: mock.Mock(shutdown=shutdown))
    monkeypatch.setattr(source, "run", mock.AsyncMock())
    monkeypatch.setattr(source.Path, "is_file", lambda self: True)

    await asyncio.to_thread(source.main, ["--file", "x.csv"])

    shutdown.assert_called_once_with()
