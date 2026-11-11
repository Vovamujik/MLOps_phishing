"""OpenTelemetry: трейсы и метрики по OTLP/HTTP в Collector, настройка только из OTEL_*.

Адресов в коде нет: endpoint, протокол, семплер и resource-атрибуты SDK читает из
переменных окружения сам (OTEL_EXPORTER_OTLP_ENDPOINT, OTEL_TRACES_SAMPLER,
OTEL_RESOURCE_ATTRIBUTES…). OTEL_SDK_DISABLED=true выключает всё: ни провайдеров, ни
фоновых потоков, инструментация пишет в no-op.

Resource: service.name из OTEL_SERVICE_NAME (по умолчанию — имя процесса: api, worker,
source), service.version — версия пакета, service.instance.id — hostname контейнера,
если не задан в OTEL_RESOURCE_ATTRIBUTES: у реплик `--scale worker=2` он разный, и в
трейсах видно, какой воркер взял задачу.

Семплер. Стенд: parentbased_traceidratio с аргументом 1.0 — сохраняется каждый трейс,
чтобы по любому task_id найти его трейс. Прод при сотнях задач в секунду:
parentbased_traceidratio 0.01–0.1 (решение принимает начало трейса — API или источник,
воркер его наследует через traceparent, трейсы не рвутся) плюс tail sampling в
Collector: держать все трейсы с ошибкой и медленные (latency > SLO), остальное — по доле.
Head-семплинг один терял бы большую часть ошибок, tail один — грузил бы Collector всем
потоком спанов.

Метрики — push тем же OTLP раз в OTEL_METRIC_EXPORT_INTERVAL: источник живёт минуты,
pull-скрейп его бы не застал, а реплики воркера не нужно регистрировать в Prometheus.
Границы гистограмм заданы явно (Views): API — вокруг задержки публикации и её таймаута
2 с, inference — вокруг INFERENCE_DELAY_SECONDS из настроек, задача целиком — от
долей секунды до минут ожидания в очереди. Память и CPU процесса — system-metrics,
только process.memory.usage и process.cpu.time, без десятков системных метрик хоста.

Collector недоступен: BatchSpanProcessor копит спаны в очереди (до 2048, лишнее
отбрасывает) и отправляет из фонового потока — запросы приложения не ждут экспорта;
метрики копятся в памяти (кумулятивно) и уйдут, когда Collector вернётся.
Ошибка экспорта логируется не чаще раза в минуту (ExportLogThrottle), иначе каждый
батч давал бы WARNING. При выходе shutdown отправляет остаток с коротким таймаутом.
"""

import logging
import os
import socket
import time
from dataclasses import dataclass, field

from opentelemetry import metrics, trace
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.asyncpg import AsyncPGInstrumentor
from opentelemetry.instrumentation.system_metrics import SystemMetricsInstrumentor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import MetricReader, PeriodicExportingMetricReader
from opentelemetry.sdk.metrics.view import DropAggregation, ExplicitBucketHistogramAggregation, View
from opentelemetry.sdk.resources import (
    SERVICE_INSTANCE_ID,
    SERVICE_NAME,
    SERVICE_VERSION,
    OTELResourceDetector,
    Resource,
)
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from mlops_phishing.metadata import get_version

log = logging.getLogger(__name__)

# Короткоживущему источнику не ждать мёртвый Collector при выходе дольше этого
SHUTDOWN_TIMEOUT_MS = 3000
# Логгеры экспорта: при лежащем Collector пишут на каждый батч
EXPORT_LOGGERS = ("opentelemetry.exporter", "opentelemetry.sdk")
EXPORT_LOG_INTERVAL = 60.0

TRACER_NAME = "mlops_phishing"
SEMCONV_OPT_IN = "OTEL_SEMCONV_STABILITY_OPT_IN"

# Ответ API — публикация с подтверждением: миллисекунды, потолок — PUBLISH_TIMEOUT 2 с
HTTP_DURATION_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 2.5, 5.0)
# Inference — доли от задержки модели: густо около неё, чтобы p95/p99 не слипались
INFERENCE_SHARES = (0.25, 0.5, 0.75, 1.0, 1.025, 1.05, 1.1, 1.15, 1.25, 1.5, 2.0, 3.0, 5.0, 10.0)
# Без задержки остаётся только модель: миллисекунды
INFERENCE_FALLBACK_BUCKETS = (0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0)
# Задача целиком — с ожиданием в очереди: при полной очереди это секунды и десятки секунд
TASK_DURATION_BUCKETS = (0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.5, 10.0, 15.0, 20.0, 30.0, 60.0, 120.0)
# Метрики, которые дублируют наши или не нужны дашборду: лишние ряды без пользы
DROPPED_METRICS = (
    "messaging.publish.duration",
    "messaging.process.duration",
    "http.server.request.body.size",
    "http.server.response.body.size",
)
# Память и CPU процесса; остальное system-metrics (диски, сеть хоста, gc) не нужно
PROCESS_METRICS: dict[str, list[str] | None] = {
    "process.memory.usage": None,
    "process.cpu.time": ["user", "system"],
}


def is_disabled() -> bool:
    return os.environ.get("OTEL_SDK_DISABLED", "").strip().lower() == "true"


def build_resource(service: str) -> Resource:
    """Атрибуты по умолчанию; заданное в OTEL_SERVICE_NAME и OTEL_RESOURCE_ATTRIBUTES важнее."""
    defaults = {
        SERVICE_NAME: service,
        SERVICE_VERSION: get_version(),
        SERVICE_INSTANCE_ID: socket.gethostname(),
    }
    return Resource.create(defaults).merge(OTELResourceDetector().detect())


def inference_buckets(delay: float) -> tuple[float, ...]:
    """Границы гистограммы inference под задержку модели из настроек."""
    if delay <= 0:
        return INFERENCE_FALLBACK_BUCKETS
    return tuple(round(delay * share, 4) for share in INFERENCE_SHARES)


def metric_views(inference_delay: float) -> list[View]:
    histograms = {
        "http.server.request.duration": HTTP_DURATION_BUCKETS,
        "worker.inference.duration": inference_buckets(inference_delay),
        "worker.task.duration": TASK_DURATION_BUCKETS,
    }
    views = [
        View(instrument_name=name, aggregation=ExplicitBucketHistogramAggregation(bounds))
        for name, bounds in histograms.items()
    ]
    views += [View(instrument_name=name, aggregation=DropAggregation()) for name in DROPPED_METRICS]
    return views


def build_meter_provider(
    resource: Resource, readers: list[MetricReader], inference_delay: float
) -> MeterProvider:
    """MeterProvider с нашими границами бакетов; читатели — OTLP в проде, память в тестах."""
    return MeterProvider(
        resource=resource, metric_readers=readers, views=metric_views(inference_delay)
    )


class ExportLogThrottle(logging.Filter):
    """Не больше одной записи экспортёра в минуту на логгер: Collector лёг — видно, но не шумно."""

    def __init__(self, interval: float = EXPORT_LOG_INTERVAL) -> None:
        super().__init__()
        self.interval = interval
        self._last: dict[str, float] = {}

    def filter(self, record: logging.LogRecord) -> bool:
        if not record.name.startswith(EXPORT_LOGGERS):
            return True
        now = time.monotonic()
        last = self._last.get(record.name)
        if last is not None and now - last < self.interval:
            return False
        self._last[record.name] = now
        return True


@dataclass
class Telemetry:
    """Провайдеры, созданные этим вызовом: их нужно сбросить и закрыть при выходе.

    Пусто, если SDK выключен или провайдер уже настроен раньше (второй create_app,
    тесты с in-memory экспортёром): закрывать чужой провайдер нельзя.
    """

    owned: list[TracerProvider | MeterProvider] = field(default_factory=list)

    def shutdown(self) -> None:
        """Отправить остаток и остановить фоновые потоки. Повторный вызов безопасен."""
        # shutdown сам отправляет остаток: спаны из очереди, последний срез метрик —
        # короткий источник успевает отчитаться. Каждый экспорт ограничен таймаутом
        for provider in self.owned:
            provider.shutdown()
        self.owned.clear()


def setup_telemetry(
    service: str, *, inference_delay: float = 0.0, instrument_asyncpg: bool = False
) -> Telemetry:
    """Включить трейсы и метрики процесса. Идемпотентно: второй вызов вернёт настроенное.

    inference_delay: задержка модели из настроек — под неё границы гистограммы inference.
    instrument_asyncpg: спаны на каждый запрос к Postgres (текст запроса без параметров).
    """
    # Стабильные HTTP-конвенции: метрика http.server.request.duration в секундах и атрибуты
    # http.response.status_code, http.route. Без флага инструментация пишет устаревшие
    # http.server.duration в миллисекундах. Читается один раз, до инструментации FastAPI
    os.environ.setdefault(SEMCONV_OPT_IN, "http")
    current = trace.get_tracer_provider()
    if isinstance(current, TracerProvider):
        if instrument_asyncpg:
            enable_asyncpg_spans()
        return Telemetry()
    if is_disabled():
        log.info("OpenTelemetry выключен (OTEL_SDK_DISABLED)")
        return Telemetry()

    for handler in logging.getLogger().handlers:
        handler.addFilter(ExportLogThrottle())
    resource = build_resource(service)
    # Семплер не передаём: TracerProvider берёт его из OTEL_TRACES_SAMPLER(_ARG)
    tracer_provider = TracerProvider(resource=resource)
    tracer_provider.add_span_processor(
        BatchSpanProcessor(OTLPSpanExporter(timeout=SHUTDOWN_TIMEOUT_MS / 1000))
    )
    trace.set_tracer_provider(tracer_provider)
    # Интервал экспорта — из OTEL_METRIC_EXPORT_INTERVAL; временность по умолчанию
    # кумулятивная — её и ждёт OTLP-приёмник Prometheus
    reader = PeriodicExportingMetricReader(OTLPMetricExporter(timeout=SHUTDOWN_TIMEOUT_MS / 1000))
    meter_provider = build_meter_provider(resource, [reader], inference_delay)
    metrics.set_meter_provider(meter_provider)
    SystemMetricsInstrumentor(config=PROCESS_METRICS).instrument(meter_provider=meter_provider)
    if instrument_asyncpg:
        enable_asyncpg_spans()
    log.info(
        "OpenTelemetry включён: %s",
        resource.attributes.get(SERVICE_NAME),
        extra={"service_instance_id": resource.attributes.get(SERVICE_INSTANCE_ID)},
    )
    return Telemetry(owned=[tracer_provider, meter_provider])


def enable_asyncpg_spans() -> None:
    instrumentor = AsyncPGInstrumentor()
    if not instrumentor.is_instrumented_by_opentelemetry:
        instrumentor.instrument()


def tracer() -> trace.Tracer:
    """Трейсер ручных спанов: inference, запись в базу, ожидание источника."""
    return trace.get_tracer(TRACER_NAME, get_version())
