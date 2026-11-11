"""OpenTelemetry: трейсы по OTLP/HTTP в Collector, настройка только из OTEL_*.

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

Collector недоступен: BatchSpanProcessor копит спаны в очереди (до 2048, лишнее
отбрасывает) и отправляет из фонового потока — запросы приложения не ждут экспорта.
Ошибка экспорта логируется не чаще раза в минуту (ExportLogThrottle), иначе каждый
батч давал бы WARNING. При выходе shutdown отправляет остаток с коротким таймаутом.
"""

import logging
import os
import socket
import time
from dataclasses import dataclass, field

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.asyncpg import AsyncPGInstrumentor
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

    owned: list[TracerProvider] = field(default_factory=list)

    def shutdown(self) -> None:
        """Отправить остаток и остановить фоновые потоки. Повторный вызов безопасен."""
        for provider in self.owned:
            provider.force_flush(SHUTDOWN_TIMEOUT_MS)
            provider.shutdown()
        self.owned.clear()


def setup_telemetry(service: str, *, instrument_asyncpg: bool = False) -> Telemetry:
    """Включить трейсинг для процесса. Идемпотентно: второй вызов вернёт уже настроенное.

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
    if instrument_asyncpg:
        enable_asyncpg_spans()
    log.info(
        "OpenTelemetry включён: %s",
        resource.attributes.get(SERVICE_NAME),
        extra={"service_instance_id": resource.attributes.get(SERVICE_INSTANCE_ID)},
    )
    return Telemetry(owned=[tracer_provider])


def enable_asyncpg_spans() -> None:
    instrumentor = AsyncPGInstrumentor()
    if not instrumentor.is_instrumented_by_opentelemetry:
        instrumentor.instrument()


def tracer() -> trace.Tracer:
    """Трейсер ручных спанов: inference, запись в базу, ожидание источника."""
    return trace.get_tracer(TRACER_NAME, get_version())
