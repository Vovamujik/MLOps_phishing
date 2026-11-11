"""Тесты access-лога: уровень записи, IP клиента, связь с request_id."""

import logging

import pytest
from httpx import ASGITransport, AsyncClient

from mlops_phishing.app import QUIET_PATHS, access_log_level, create_app

ACCESS_LOGGER = "mlops_phishing.app"


@pytest.mark.parametrize(
    ("path", "status_code", "expected"),
    [
        ("/api/v1/version", 200, logging.INFO),
        ("/api/v1/version", 307, logging.INFO),
        ("/api/v1/version", 400, logging.WARNING),
        ("/does-not-exist", 404, logging.WARNING),
        ("/api/v1/version", 422, logging.WARNING),
        ("/api/v1/health", 500, logging.ERROR),
        ("/api/v1/health", 503, logging.ERROR),
        ("/healthz", 200, logging.DEBUG),
        ("/healthz", 404, logging.WARNING),
        ("/healthz", 503, logging.ERROR),
        ("/api/v1/health", 200, logging.DEBUG),
    ],
)
def test_level_depends_on_status_and_path(path, status_code, expected):
    """5xx — ERROR, 4xx — WARNING, успешный тихий путь — DEBUG, остальное — INFO."""
    assert access_log_level(path, status_code) == expected


def test_planned_load_shedding_is_a_warning_not_an_error():
    """503 с Retry-After — очередь заполнена, штатная работа под нагрузкой, а не авария."""
    assert access_log_level("/api/v1/tasks", 503, shedding=True) == logging.WARNING
    assert access_log_level("/api/v1/tasks", 503) == logging.ERROR


def test_health_probes_are_quiet():
    """Пробы дёргаются оркестратором и мониторингом постоянно и не должны шуметь на INFO."""
    assert {"/healthz", "/api/v1/health"} <= QUIET_PATHS


def access_records(json_log, method: str, path: str) -> list[dict[str, object]]:
    """Строки access-лога о запросах к конкретному пути."""
    return [
        record
        for record in json_log.records
        if record["logger"] == ACCESS_LOGGER
        and record.get("path") == path
        and record.get("method") == method
    ]


async def test_request_is_logged_once_with_client_and_request_id(client, json_log):
    """Одна строка на запрос, с IP клиента и тем же id, что в заголовке ответа."""
    response = await client.get("/api/v1/version")

    [record] = access_records(json_log, "GET", "/api/v1/version")
    assert record["level"] == "info"
    assert record["message"] == "GET /api/v1/version -> 200"
    assert record["client"] == "127.0.0.1"
    assert record["status_code"] == 200
    assert isinstance(record["duration_ms"], float)
    assert record["request_id"] == response.headers["X-Request-ID"]


async def test_client_error_is_logged_as_warning(client, json_log):
    """Запрос к несуществующему адресу виден как предупреждение."""
    await client.get("/does-not-exist")

    [record] = access_records(json_log, "GET", "/does-not-exist")
    assert record["level"] == "warning"


async def test_degraded_health_is_logged_as_error(client, json_log):
    """503 от health при недоступной базе попадает в лог как ошибка, хоть путь и тихий."""
    response = await client.get("/api/v1/health")

    assert response.status_code == 503
    [record] = access_records(json_log, "GET", "/api/v1/health")
    assert record["level"] == "error"


async def test_successful_liveness_probe_is_logged_at_debug(client, json_log):
    """Успешный /healthz уходит на DEBUG и не засоряет INFO."""
    await client.get("/healthz")

    [record] = access_records(json_log, "GET", "/healthz")
    assert record["level"] == "debug"


async def test_request_id_does_not_leak_out_of_request(client, json_log):
    await client.get("/api/v1/version")
    logging.getLogger("mlops_phishing.test").info("после запроса")

    [record] = json_log.find("после запроса")
    assert "request_id" not in record


async def test_unhandled_exception_is_logged_as_error_with_request_id(request):
    """Запрос, упавший с исключением, всё равно оставляет строку в логе со стектрейсом.

    uvicorn.access выключен, так что эта строка — единственная запись о методе,
    пути и клиенте такого запроса. Используется отдельный экземпляр приложения,
    чтобы не добавлять падающий маршрут в общее. Перехват логов — после create_app:
    он заново настраивает корневой логгер.
    """
    crashing_app = create_app()
    json_log = request.getfixturevalue("json_log")

    @crashing_app.get("/boom")
    async def boom() -> None:
        raise RuntimeError("kaboom")

    transport = ASGITransport(app=crashing_app, raise_app_exceptions=False)
    async with AsyncClient(transport=transport, base_url="http://test") as http_client:
        response = await http_client.get("/boom")

    assert response.status_code == 500
    [record] = access_records(json_log, "GET", "/boom")
    assert record["level"] == "error"
    assert record["message"] == "GET /boom -> 500"
    assert "RuntimeError: kaboom" in str(record["exception"])
    assert record["request_id"]
