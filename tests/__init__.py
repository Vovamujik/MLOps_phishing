"""Тесты приложения.

OpenTelemetry в тестах: глобальный провайдер с экспортом в память ставится здесь, до
импорта приложения (conftest импортирует `mlops_phishing.app`, а create_app включает
телеметрию). setup_telemetry видит готовый провайдер и не заводит OTLP-экспорт, а
тесты читают спаны из SPANS.
"""

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

SPANS = InMemorySpanExporter()

_tracer_provider = TracerProvider()
_tracer_provider.add_span_processor(SimpleSpanProcessor(SPANS))
trace.set_tracer_provider(_tracer_provider)
