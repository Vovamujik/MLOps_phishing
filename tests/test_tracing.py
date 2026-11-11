"""Трейсы: контекст через брокер и пул потоков, ошибки, корневые трейсы источника, trace_id в логах.

Брокер — TestRabbitBroker: он прогоняет публикацию и обработку через те же
middleware, что и настоящий, поэтому traceparent в заголовках и спан process — настоящие.
Сквозной трейс на живом RabbitMQ — в интеграционных тестах.
"""

import asyncio
import logging
import socket
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from unittest import mock

import pytest
from faststream.rabbit import TestRabbitBroker
from opentelemetry import trace
from opentelemetry.sdk.resources import SERVICE_INSTANCE_ID, SERVICE_NAME, SERVICE_VERSION
from opentelemetry.trace import SpanKind, StatusCode

from mlops_phishing import messaging, tasks_repo, telemetry
from mlops_phishing.config import Settings
from mlops_phishing.metadata import get_version
from mlops_phishing.source import Backoff, Row, SourceStats, run_batch, task_id_for
from mlops_phishing.worker import build_worker

from .conftest import FakeClassifier, FakePublisher, broker_nack, make_model, span_named


def make_task() -> messaging.TaskMessage:
    return messaging.TaskMessage(
        task_id=uuid.uuid4(),
        url="http://paypal-verify.example/login",
        source="api",
        enqueued_at=datetime.now(UTC),
    )


@pytest.fixture
def worker():
    built = build_worker(Settings(inference_delay_seconds=0, worker_pool_size=2))
    built.state.model = make_model(FakeClassifier(phishing_probability=0.9), version="4")
    built.state.executor = ThreadPoolExecutor(2)
    built.state.db_pool = mock.Mock()
    yield built
    built.state.executor.shutdown(wait=True)


async def publish_through_worker(worker, task: messaging.TaskMessage) -> None:
    """Опубликовать как API — через publish_task — и дать воркеру обработать."""
    async with TestRabbitBroker(worker.broker) as broker:
        await messaging.publish_task(broker, task)


# --- Сквозной трейс воркера ---------------------------------------------------


async def test_process_continues_publisher_trace_and_inference_is_its_child(
    worker, spans, monkeypatch
):
    """Один trace_id от издателя до записи в базу; inference — ребёнок process, а не корень."""
    monkeypatch.setattr(tasks_repo, "save_result", mock.AsyncMock(return_value=True))
    task = make_task()

    with telemetry.tracer().start_as_current_span("POST /api/v1/tasks") as root:
        await publish_through_worker(worker, task)

    publish = span_named(spans, "default.tasks publish")
    process = span_named(spans, "default.tasks process")
    inference = span_named(spans, "inference")
    db_write = span_named(spans, "db.save_result")
    trace_id = root.get_span_context().trace_id
    assert {s.context.trace_id for s in (publish, process, inference, db_write)} == {trace_id}
    assert (publish.kind, process.kind) == (SpanKind.PRODUCER, SpanKind.CONSUMER)
    assert inference.parent.span_id == process.context.span_id
    assert db_write.parent.span_id == process.context.span_id
    assert inference.attributes["model.alias"] == "champion"
    assert inference.attributes["model.version"] == "4"
    assert process.attributes["task.id"] == str(task.task_id)
    assert process.attributes["messaging.rabbitmq.delivery_count"] == 0
    assert db_write.attributes["task.id"] == str(task.task_id)


async def test_worker_logs_carry_trace_and_span_ids(worker, spans, monkeypatch, json_log):
    monkeypatch.setattr(tasks_repo, "save_result", mock.AsyncMock(return_value=True))

    await publish_through_worker(worker, make_task())

    process = span_named(spans, "default.tasks process")
    [done] = json_log.find("Задача обработана")
    assert done["trace_id"] == trace.format_trace_id(process.context.trace_id)
    assert done["span_id"] == trace.format_span_id(process.context.span_id)


async def test_inference_thread_sees_task_context(worker, spans, monkeypatch, json_log):
    """В потоке пула те же contextvars: записи оттуда — с task_id и trace_id."""
    task = make_task()
    monkeypatch.setattr(tasks_repo, "save_result", mock.AsyncMock(return_value=True))

    def logging_classifier_score(url: str) -> float:
        logging.getLogger("mlops_phishing.ml").info("inference в потоке")
        return 0.5

    worker.state.model = mock.Mock(
        name="model", version="4", alias="champion", phishing_probability=logging_classifier_score
    )
    worker.state.model.name = "phishing-url-clf"

    await publish_through_worker(worker, task)

    inference = span_named(spans, "inference")
    [record] = json_log.find("inference в потоке")
    assert record["task_id"] == str(task.task_id)
    assert record["span_id"] == trace.format_span_id(inference.context.span_id)


async def test_database_error_marks_spans_error_with_exception(worker, spans, monkeypatch):
    monkeypatch.setattr(tasks_repo, "save_result", mock.AsyncMock(side_effect=OSError("db down")))
    limit = worker.state.settings.tasks_delivery_limit
    task = make_task()

    async with TestRabbitBroker(worker.broker) as broker:
        # Последняя разрешённая попытка: после неё брокер переложит задачу в DLQ
        await broker.publish(
            task,
            queue=messaging.TASKS_QUEUE,
            correlation_id=str(task.task_id),
            headers={"x-delivery-count": limit},
        )

    db_write = span_named(spans, "db.save_result")
    process = span_named(spans, "default.tasks process")
    assert db_write.status.status_code is StatusCode.ERROR
    [exception] = [e for e in db_write.events if e.name == "exception"]
    assert exception.attributes["exception.type"] == "OSError"
    assert process.status.status_code is StatusCode.ERROR
    assert process.attributes["messaging.rabbitmq.delivery_count"] == limit
    assert process.attributes["messaging.delivery.attempt"] == limit + 1
    assert process.attributes["messaging.rabbitmq.dead_lettered"] is True
    [dlq] = [e for e in process.events if e.name == "dead_lettered"]
    assert dlq.attributes["reason"] == "исчерпаны попытки"


async def test_malformed_message_trace_shows_dlq(worker, spans):
    async with TestRabbitBroker(worker.broker) as broker:
        with pytest.raises(Exception, match="validation"):
            await broker.publish({"url": "no task id"}, queue=messaging.TASKS_QUEUE)

    process = span_named(spans, "default.tasks process")
    assert process.status.status_code is StatusCode.ERROR
    assert [e.name for e in process.events] == ["dead_lettered", "exception"]


# --- API: 503 тоже даёт трейс -------------------------------------------------


async def test_503_trace_has_error_status_and_broker_exception(client, install_publisher, spans):
    install_publisher(FakePublisher(error=broker_nack()))

    response = await client.post("/api/v1/tasks", json={"url": "http://a.example/"})

    assert response.status_code == 503
    server = span_named(spans, "POST /api/v1/tasks")
    assert server.kind is SpanKind.SERVER
    assert server.status.status_code is StatusCode.ERROR
    [exception] = [e for e in server.events if e.name == "exception"]
    assert str(exception.attributes["exception.type"]).endswith("DeliveryError")
    assert uuid.UUID(str(server.attributes["task.id"]))
    assert server.attributes["http.response.status_code"] == 503


async def test_health_probes_are_not_traced(client, spans):
    await client.get("/healthz")
    await client.get("/api/v1/health")
    await client.get("/api/v1/version")

    names = {span.name for span in spans.get_finished_spans() if span.kind is SpanKind.SERVER}
    assert names == {"GET /api/v1/version"}


# --- Источник: свой трейс на каждую задачу ------------------------------------


async def test_source_starts_a_root_trace_per_task_with_backoff_events(spans):
    rejected_once = {"left": 1}

    async def publish(task: messaging.TaskMessage) -> None:
        with telemetry.tracer().start_as_current_span("default.tasks publish"):
            if rejected_once["left"]:
                rejected_once["left"] -= 1
                raise broker_nack()

    rows = [Row(1, "http://a.example/"), Row(2, "http://b.example/")]
    with telemetry.tracer().start_as_current_span("batch"):
        await asyncio.wait_for(
            run_batch(rows, publish, "b", Backoff(0.001, 0.002), asyncio.Event(), SourceStats()),
            timeout=5,
        )

    roots = [s for s in spans.get_finished_spans() if s.name == "source.task"]
    assert [r.attributes["task.id"] for r in roots] == [
        str(task_id_for("b", 1)),
        str(task_id_for("b", 2)),
    ]
    # Не дети спана партии и не один трейс на всех
    assert all(r.parent is None for r in roots)
    assert len({r.context.trace_id for r in roots}) == 2
    first = roots[0]
    [rejected] = [e for e in first.events if e.name == "broker.rejected"]
    assert rejected.attributes["attempt"] == 1
    assert rejected.attributes["delay_s"] > 0
    assert first.attributes["source.attempts"] == 2
    backoff = span_named(spans, "source.backoff")
    assert backoff.parent.span_id == first.context.span_id
    assert backoff.attributes["attempt"] == 1
    publishes = [s for s in spans.get_finished_spans() if s.name == "default.tasks publish"]
    assert {p.context.trace_id for p in publishes} == {r.context.trace_id for r in roots}


# --- Настройка SDK ----------------------------------------------------------


def test_resource_defaults_and_env_overrides(monkeypatch):
    monkeypatch.delenv("OTEL_SERVICE_NAME", raising=False)
    monkeypatch.setenv("OTEL_RESOURCE_ATTRIBUTES", "deployment.environment.name=staging")

    resource = telemetry.build_resource("worker")

    attributes = resource.attributes
    assert attributes[SERVICE_NAME] == "worker"
    assert attributes[SERVICE_VERSION] == get_version()
    assert attributes[SERVICE_INSTANCE_ID] == socket.gethostname()
    assert attributes["deployment.environment.name"] == "staging"

    monkeypatch.setenv("OTEL_SERVICE_NAME", "worker-blue")
    monkeypatch.setenv("OTEL_RESOURCE_ATTRIBUTES", "service.instance.id=worker-1")
    overridden = telemetry.build_resource("worker").attributes
    assert (overridden[SERVICE_NAME], overridden[SERVICE_INSTANCE_ID]) == (
        "worker-blue",
        "worker-1",
    )


def test_setup_reuses_configured_provider_and_owns_nothing():
    """Второй вызов (или тесты с провайдером в памяти) не заводит экспорт и не закрывает чужое."""
    tracing = telemetry.setup_telemetry("api")

    assert tracing.owned == []
    tracing.shutdown()


def test_export_errors_are_throttled():
    throttle = telemetry.ExportLogThrottle(interval=60)

    def record(name: str) -> logging.LogRecord:
        return logging.LogRecord(name, logging.WARNING, __file__, 1, "export failed", (), None)

    exporter = "opentelemetry.exporter.otlp.proto.http.trace_exporter"
    assert throttle.filter(record(exporter)) is True
    assert throttle.filter(record(exporter)) is False
    assert throttle.filter(record("mlops_phishing.worker")) is True
    assert throttle.filter(record("mlops_phishing.worker")) is True


@pytest.fixture
def fresh_otel(monkeypatch):
    """Процесс, где OpenTelemetry ещё не настроен; установленный провайдер — в список."""
    installed: list[object] = []
    monkeypatch.setattr(telemetry.trace, "get_tracer_provider", trace.ProxyTracerProvider)
    monkeypatch.setattr(telemetry.trace, "set_tracer_provider", installed.append)
    monkeypatch.setattr(telemetry.metrics, "set_meter_provider", installed.append)
    monkeypatch.setattr(telemetry, "SystemMetricsInstrumentor", mock.Mock())
    monkeypatch.setattr(logging.getLogger(), "handlers", [logging.StreamHandler()])
    return installed


def test_disabled_sdk_installs_nothing(fresh_otel, monkeypatch):
    monkeypatch.setenv("OTEL_SDK_DISABLED", "true")

    tracing = telemetry.setup_telemetry("source")

    assert (tracing.owned, fresh_otel) == ([], [])


def test_dead_collector_does_not_block_exit(fresh_otel, monkeypatch):
    """Collector не запущен: спаны копятся в фоне, выход ждёт не дольше таймаута."""
    monkeypatch.delenv("OTEL_SDK_DISABLED", raising=False)
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://127.0.0.1:59996")

    tracing = telemetry.setup_telemetry("source")
    provider, meter_provider = fresh_otel
    assert tracing.owned == [provider, meter_provider]
    # Память и CPU процесса — только они, без системных метрик хоста
    telemetry.SystemMetricsInstrumentor.assert_called_once_with(config=telemetry.PROCESS_METRICS)
    assert [type(f) for f in logging.getLogger().handlers[0].filters] == [
        telemetry.ExportLogThrottle
    ]
    tracer = provider.get_tracer("test")
    started = time.perf_counter()
    for _ in range(100):
        with tracer.start_as_current_span("task"):
            pass
    assert time.perf_counter() - started < 0.5

    tracing.shutdown()
    tracing.shutdown()

    # Трейсы и последний срез метрик — каждый не дольше таймаута экспорта
    assert time.perf_counter() - started < 2 * telemetry.SHUTDOWN_TIMEOUT_MS / 1000 + 1
