"""Топология RabbitMQ: очередь задач, DLX и DLQ, policy с лимитами.

Лимит длины и число доставок заданы policy, а не аргументами очереди: policy
меняется одной командой на работающей очереди, без пересоздания и потери задач.
Все запросы идемпотентны, повторный запуск ничего не ломает.

Запуск: python -m mlops_phishing.rabbit_setup
"""

import base64
import json
import logging
import time
import urllib.error
import urllib.request
from urllib.parse import quote, unquote, urlsplit

from mlops_phishing.config import Settings, get_settings
from mlops_phishing.messaging import (
    DEAD_LETTER_EXCHANGE,
    DEAD_LETTER_QUEUE_NAME,
    TASKS_QUEUE_NAME,
)

log = logging.getLogger(__name__)

POLICY_NAME = "tasks-limits"
QUORUM = {"x-queue-type": "quorum"}


def tasks_policy(settings: Settings) -> dict[str, object]:
    """Лимит готовых сообщений, отказ публикации сверх него, повторы и DLQ."""
    return {
        "pattern": f"^{TASKS_QUEUE_NAME}$",
        "apply-to": "queues",
        "priority": 0,
        "definition": {
            "max-length": settings.tasks_queue_max_length,
            # Сверх лимита брокер отвечает издателю nack: это и есть сигнал противодавления.
            # Quorum-очередь включает отказ, когда готовых стало больше лимита (в UI видно
            # max-length + 1), и снова принимает, когда их 80% лимита
            "overflow": "reject-publish",
            "delivery-limit": settings.tasks_delivery_limit,
            "dead-letter-exchange": DEAD_LETTER_EXCHANGE,
            "dead-letter-routing-key": DEAD_LETTER_QUEUE_NAME,
        },
    }


def topology_requests(settings: Settings) -> list[tuple[str, str, dict[str, object]]]:
    """Запросы к HTTP API: (метод, путь, тело). DLQ создаётся до основной очереди."""
    vhost = quote(urlsplit(settings.rabbitmq_url).path[1:] or "/", safe="")
    durable_quorum = {"durable": True, "arguments": QUORUM}
    return [
        ("PUT", f"exchanges/{vhost}/{DEAD_LETTER_EXCHANGE}", {"type": "direct", "durable": True}),
        ("PUT", f"queues/{vhost}/{DEAD_LETTER_QUEUE_NAME}", durable_quorum),
        (
            "POST",
            f"bindings/{vhost}/e/{DEAD_LETTER_EXCHANGE}/q/{DEAD_LETTER_QUEUE_NAME}",
            {"routing_key": DEAD_LETTER_QUEUE_NAME},
        ),
        ("PUT", f"policies/{vhost}/{POLICY_NAME}", tasks_policy(settings)),
        ("PUT", f"queues/{vhost}/{TASKS_QUEUE_NAME}", durable_quorum),
    ]


def _send(settings: Settings, method: str, path: str, body: dict[str, object]) -> None:
    amqp = urlsplit(settings.rabbitmq_url)
    credentials = f"{unquote(amqp.username or '')}:{unquote(amqp.password or '')}"
    request = urllib.request.Request(  # noqa: S310 — адрес из настроек, схема http(s)
        f"{settings.rabbitmq_management_url.rstrip('/')}/api/{path}",
        data=json.dumps(body).encode(),
        method=method,
        headers={
            "Content-Type": "application/json",
            "Authorization": "Basic " + base64.b64encode(credentials.encode()).decode(),
        },
    )
    with urllib.request.urlopen(request, timeout=10):  # noqa: S310
        pass


def apply_topology(settings: Settings, attempts: int = 30) -> None:
    """Применить топологию. Ждёт HTTP API: плагин управления стартует позже брокера."""
    for method, path, body in topology_requests(settings):
        for attempt in range(1, attempts + 1):
            try:
                _send(settings, method, path, body)
                break
            except (urllib.error.URLError, ConnectionError) as exc:
                # HTTPError с кодом 4xx — ошибка конфигурации, ждать бессмысленно
                if isinstance(exc, urllib.error.HTTPError) and exc.code < 500:
                    raise
                if attempt == attempts:
                    raise
                time.sleep(1)
        log.info("%s %s", method, path)
    log.info(
        "Топология готова: %s (лимит %d, повторов %d) → %s",
        TASKS_QUEUE_NAME,
        settings.tasks_queue_max_length,
        settings.tasks_delivery_limit,
        DEAD_LETTER_QUEUE_NAME,
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    apply_topology(get_settings())


if __name__ == "__main__":
    main()
