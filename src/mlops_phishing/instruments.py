"""Метрики конвейера: то, что в прошлой работе мерилось руками, теперь считается само.

Имена — OpenTelemetry; Prometheus получает их через OTLP в своём виде: точки → `_`,
единица в суффикс, у счётчиков `_total` (`source.backoff.duration`, s →
`source_backoff_duration_seconds_total`). Единицы в фигурных скобках ({task})
Prometheus отбрасывает.

Лейблы только низкой кардинальности: у worker.tasks.processed — result ∈ ok|error|dlq,
у остальных нет своих. Экземпляр (job/instance) Prometheus берёт из resource, а task_id
и trace_id остаются в логах и трейсах — в метриках каждая задача стала бы рядом.

Инструменты создаются на глобальном meter при импорте: пока провайдер не настроен,
это прокси, после setup_telemetry они пишут в настоящий.
"""

from collections.abc import Iterable
from enum import StrEnum

from opentelemetry import metrics
from opentelemetry.metrics import CallbackOptions, Observation

from mlops_phishing.metadata import get_version

_meter = metrics.get_meter("mlops_phishing", get_version())


class Outcome(StrEnum):
    """Исход доставки: ok — записано (или уже было), error — вернётся в очередь, dlq — в DLQ."""

    OK = "ok"
    ERROR = "error"
    DLQ = "dlq"


# --- Источник ---------------------------------------------------------------

source_published = _meter.create_counter(
    "source.tasks.published", unit="{task}", description="Задач, подтверждённых брокером."
)
source_rejections = _meter.create_counter(
    "source.broker.rejections",
    unit="{rejection}",
    description="Отказов брокера (nack при reject-publish): очередь была полна.",
)
source_backoff = _meter.create_counter(
    "source.backoff.duration", unit="s", description="Суммарная пауза источника после отказов."
)

# --- Воркер -----------------------------------------------------------------

worker_processed = _meter.create_counter(
    "worker.tasks.processed", unit="{task}", description="Доставок задач по исходу (result)."
)
worker_inference = _meter.create_histogram(
    "worker.inference.duration", unit="s", description="Время inference в потоке пула."
)
worker_task_duration = _meter.create_histogram(
    "worker.task.duration",
    unit="s",
    description="От постановки в очередь (enqueued_at) до записи результата: для SLO.",
)
worker_in_flight = _meter.create_up_down_counter(
    "worker.tasks.in_flight", unit="{task}", description="Задач в работе прямо сейчас."
)
worker_redeliveries = _meter.create_counter(
    "worker.redeliveries", unit="{delivery}", description="Повторных доставок задач."
)

# --- API --------------------------------------------------------------------

_queue_limit: dict[str, int] = {}


def _observe_queue_limit(options: CallbackOptions) -> Iterable[Observation]:
    if limit := _queue_limit.get("value"):
        yield Observation(limit)


def observe_queue_limit(limit: int) -> None:
    """Лимит очереди из настроек — линия на графике глубины и порог алерта.

    Гауж регистрируется один раз на процесс; 0 (очередь без лимита) не публикуется.
    """
    if "value" not in _queue_limit:
        _meter.create_observable_gauge(
            "tasks.queue.limit",
            callbacks=[_observe_queue_limit],
            unit="{message}",
            description="Лимит готовых сообщений в очереди tasks (policy max-length).",
        )
    _queue_limit["value"] = limit
