"""Тесты топологии RabbitMQ: что и в каком порядке уходит в HTTP API."""

import base64
import json
import urllib.error

import pytest

from mlops_phishing import rabbit_setup
from mlops_phishing.config import Settings


def test_policy_limits_queue_and_routes_failures_to_dlq():
    """Лимит и reject-publish дают сигнал источнику, delivery-limit + DLX — путь в DLQ."""
    policy = rabbit_setup.tasks_policy(Settings(tasks_queue_max_length=50, tasks_delivery_limit=2))

    assert policy["pattern"] == "^tasks$"
    assert policy["definition"] == {
        "max-length": 50,
        "overflow": "reject-publish",
        "delivery-limit": 2,
        "dead-letter-exchange": "tasks.dlx",
        "dead-letter-routing-key": "tasks.dlq",
    }


def test_zero_limit_leaves_queue_unbounded_but_keeps_dlq():
    """Режим эксперимента «без ограничений»: нет лимита и nack, повторы и DLQ на месте."""
    policy = rabbit_setup.tasks_policy(Settings(tasks_queue_max_length=0, tasks_delivery_limit=2))

    assert policy["definition"] == {
        "delivery-limit": 2,
        "dead-letter-exchange": "tasks.dlx",
        "dead-letter-routing-key": "tasks.dlq",
    }


def test_dlq_exists_before_the_queue_that_dead_letters_into_it():
    """Иначе первое же «мёртвое» сообщение некуда было бы переложить."""
    paths = [path for _, path, _ in rabbit_setup.topology_requests(Settings())]

    assert paths == [
        "exchanges/%2F/tasks.dlx",
        "queues/%2F/tasks.dlq",
        "bindings/%2F/e/tasks.dlx/q/tasks.dlq",
        "policies/%2F/tasks-limits",
        "queues/%2F/tasks",
    ]


def test_vhost_is_taken_from_amqp_url():
    requests = rabbit_setup.topology_requests(Settings(rabbitmq_url="amqp://u:p@h:5672/lab"))

    assert requests[0][1] == "exchanges/lab/tasks.dlx"


class Recorder:
    """Подмена urlopen: запоминает запросы, первые `failures` раз падает."""

    def __init__(self, failures: int = 0, error: Exception | None = None) -> None:
        self.failures = failures
        self.error = error or urllib.error.URLError("connection refused")
        self.requests: list = []

    def __call__(self, request, timeout):
        self.requests.append(request)
        if self.failures:
            self.failures -= 1
            raise self.error
        return self

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@pytest.fixture
def no_sleep(monkeypatch):
    monkeypatch.setattr(rabbit_setup.time, "sleep", lambda seconds: None)


def test_requests_carry_json_and_basic_auth_from_amqp_url(monkeypatch, no_sleep):
    recorder = Recorder()
    monkeypatch.setattr(rabbit_setup.urllib.request, "urlopen", recorder)

    rabbit_setup.apply_topology(
        Settings(
            rabbitmq_url="amqp://lab:s3cr%40t@rabbit:5672/",
            rabbitmq_management_url="http://rabbit:15672/",
        )
    )

    first = recorder.requests[0]
    assert first.full_url == "http://rabbit:15672/api/exchanges/%2F/tasks.dlx"
    assert first.get_method() == "PUT"
    assert json.loads(first.data) == {"type": "direct", "durable": True}
    token = first.get_header("Authorization").removeprefix("Basic ")
    assert base64.b64decode(token).decode() == "lab:s3cr@t"
    assert len(recorder.requests) == 5


def test_waits_for_management_api_to_come_up(monkeypatch, no_sleep):
    """Плагин управления поднимается позже брокера: отказ соединения — повод подождать."""
    recorder = Recorder(failures=3)
    monkeypatch.setattr(rabbit_setup.urllib.request, "urlopen", recorder)

    rabbit_setup.apply_topology(Settings())

    assert len(recorder.requests) == 3 + 5


def test_client_error_is_not_retried(monkeypatch, no_sleep):
    """401 или 400 — ошибка конфигурации: ждать бесполезно, падаем сразу."""
    error = urllib.error.HTTPError("http://x", 401, "Unauthorized", {}, None)
    recorder = Recorder(failures=1, error=error)
    monkeypatch.setattr(rabbit_setup.urllib.request, "urlopen", recorder)

    with pytest.raises(urllib.error.HTTPError):
        rabbit_setup.apply_topology(Settings())

    assert len(recorder.requests) == 1


def test_gives_up_after_attempts(monkeypatch, no_sleep):
    monkeypatch.setattr(rabbit_setup.urllib.request, "urlopen", Recorder(failures=10))

    with pytest.raises(urllib.error.URLError):
        rabbit_setup.apply_topology(Settings(), attempts=3)
