"""Тесты access-лога: уровень записи, IP клиента, связь с request_id."""

import logging

import pytest
from httpx import ASGITransport, AsyncClient

from mlops_phishing.app import QUIET_PATHS, access_log_level, create_app
from mlops_phishing.logging_config import NO_REQUEST_ID, RequestIdFilter, get_request_id

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
    ],
)
def test_level_depends_on_status_and_path(path, status_code, expected):
    """5xx — ERROR, 4xx — WARNING, успешный тихий путь — DEBUG, остальное — INFO."""
    assert access_log_level(path, status_code) == expected


def test_liveness_probe_is_quiet():
    """Liveness дёргается оркестратором постоянно и не должен шуметь на INFO."""
    assert "/healthz" in QUIET_PATHS


@pytest.fixture
def access_log(caplog: pytest.LogCaptureFixture) -> pytest.LogCaptureFixture:
    """Перехватывать записи access-лога, включая DEBUG.

    Уровень логгера middleware временно опускается до DEBUG, иначе записи
    о /healthz даже не создаются. caplog сам вернёт уровень после теста.
    """
    caplog.set_level(logging.DEBUG, logger=ACCESS_LOGGER)
    return caplog


def records_for(caplog: pytest.LogCaptureFixture, path: str) -> list[logging.LogRecord]:
    """Записи access-лога о запросах к конкретному пути."""
    return [
        record
        for record in caplog.records
        if record.name == ACCESS_LOGGER and record.getMessage().startswith(f"GET {path} ")
    ]


async def test_request_is_logged_once_with_client_and_request_id(client, access_log):
    """Одна строка на запрос, с IP клиента и тем же id, что в заголовке ответа."""
    response = await client.get("/api/v1/version")

    records = records_for(access_log, "/api/v1/version")
    assert len(records) == 1
    [record] = records
    assert record.levelno == logging.INFO
    assert "client=127.0.0.1" in record.getMessage()
    assert get_request_id(record) == response.headers["X-Request-ID"]


async def test_client_error_is_logged_as_warning(client, access_log):
    """Запрос к несуществующему адресу виден как предупреждение."""
    await client.get("/does-not-exist")

    [record] = records_for(access_log, "/does-not-exist")
    assert record.levelno == logging.WARNING


async def test_degraded_health_is_logged_as_error(client, access_log):
    """503 от health при недоступной базе попадает в лог как ошибка."""
    response = await client.get("/api/v1/health")

    assert response.status_code == 503
    [record] = records_for(access_log, "/api/v1/health")
    assert record.levelno == logging.ERROR


async def test_successful_liveness_probe_is_logged_at_debug(client, access_log):
    """Успешный /healthz уходит на DEBUG и не засоряет INFO."""
    await client.get("/healthz")

    [record] = records_for(access_log, "/healthz")
    assert record.levelno == logging.DEBUG


async def test_unhandled_exception_is_logged_as_error_with_request_id(caplog):
    """Запрос, упавший с исключением, всё равно оставляет строку в логе.

    uvicorn.access выключен, так что эта строка — единственная запись о методе,
    пути и клиенте такого запроса. Используется отдельный экземпляр приложения,
    чтобы не добавлять падающий маршрут в общее. Фильтр request_id вешается
    на обработчик caplog: lifespan здесь не запускается, и обработчика
    приложения на корневом логгере нет.
    """
    crashing_app = create_app()

    @crashing_app.get("/boom")
    async def boom() -> None:
        raise RuntimeError("kaboom")

    caplog.set_level(logging.DEBUG, logger=ACCESS_LOGGER)
    request_id_filter = RequestIdFilter()
    caplog.handler.addFilter(request_id_filter)
    try:
        transport = ASGITransport(app=crashing_app, raise_app_exceptions=False)
        async with AsyncClient(transport=transport, base_url="http://test") as http_client:
            response = await http_client.get("/boom")
    finally:
        caplog.handler.removeFilter(request_id_filter)

    assert response.status_code == 500
    [record] = records_for(caplog, "/boom")
    assert record.levelno == logging.ERROR
    assert "-> 500" in record.getMessage()
    assert "exception=RuntimeError" in record.getMessage()
    assert get_request_id(record) != NO_REQUEST_ID
