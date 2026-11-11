"""Тесты приложения.

OpenTelemetry в тестах: глобальные провайдеры с экспортом в память ставятся здесь, до
импорта приложения (conftest импортирует `mlops_phishing.app`, а create_app включает
телеметрию). setup_telemetry видит готовый провайдер и не заводит OTLP-экспорт, а
тесты читают спаны из SPANS и метрики из METRICS. Границы гистограмм — те же Views,
что в приложении, с задержкой inference по умолчанию.
"""

from opentelemetry import metrics, trace
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from mlops_phishing.config import Settings
from mlops_phishing.telemetry import build_meter_provider

SPANS = InMemorySpanExporter()
METRICS = InMemoryMetricReader()

_tracer_provider = TracerProvider()
_tracer_provider.add_span_processor(SimpleSpanProcessor(SPANS))
trace.set_tracer_provider(_tracer_provider)

_inference_delay = Settings.model_fields["inference_delay_seconds"].default
metrics.set_meter_provider(build_meter_provider(Resource.create(), [METRICS], _inference_delay))
