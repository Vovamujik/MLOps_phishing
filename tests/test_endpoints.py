"""Тесты HTTP-контракта всех трёх эндпоинтов."""

from mlops_phishing.metadata import UNKNOWN

from .conftest import FakePool, read_pyproject_version

# --- GET /healthz --------------------------------------------------------


async def test_healthz_is_served_without_api_prefix(client):
    """Liveness живёт вне /api/v1: его адрес не меняется между версиями API."""
    assert (await client.get("/healthz")).status_code == 200
    assert (await client.get("/api/v1/healthz")).status_code == 404


async def test_healthz_returns_ok(client):
    """Быстрый ответ о живости процесса."""
    response = await client.get("/healthz")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


async def test_healthz_never_touches_the_database(client, install_pool):
    """Liveness не ходит в зависимости.

    Иначе при аварии базы оркестратор начнёт перезапускать здоровые поды,
    превращая сбой зависимости в отказ всего сервиса.
    """
    pool = install_pool(FakePool(error=AssertionError("liveness полез в базу")))

    response = await client.get("/healthz")

    assert response.status_code == 200
    assert pool.queries == []


# --- GET /api/v1/version -------------------------------------------------


async def test_version_matches_pyproject(client):
    """Эндпоинт отдаёт ровно ту версию, что записана в pyproject.toml.

    Защита от ловушки с двумя источниками правды: захардкоженная где-либо
    версия немедленно разойдётся с этим значением и покрасит тест.
    """
    body = (await client.get("/api/v1/version")).json()

    assert body["version"] == read_pyproject_version()


async def test_version_returns_full_build_identity(client):
    """Кроме версии нужны коммит и время сборки — иначе неясно, что запущено."""
    response = await client.get("/api/v1/version")
    body = response.json()

    assert response.status_code == 200
    assert body["name"] == "mlops-phishing"
    assert set(body) == {"name", "version", "git_commit", "build_time"}


async def test_version_reports_build_metadata_from_environment(client, monkeypatch):
    """Коммит и время сборки приходят из окружения образа."""
    monkeypatch.setenv("GIT_COMMIT", "deadbee")
    monkeypatch.setenv("BUILD_TIME", "2026-09-14T18:00:00Z")

    body = (await client.get("/api/v1/version")).json()

    assert body["git_commit"] == "deadbee"
    assert body["build_time"] == "2026-09-14T18:00:00Z"


async def test_version_does_not_fail_without_build_metadata(client, monkeypatch):
    """Вне собранного образа эндпоинт отвечает, а не падает."""
    monkeypatch.delenv("GIT_COMMIT", raising=False)
    monkeypatch.delenv("BUILD_TIME", raising=False)

    body = (await client.get("/api/v1/version")).json()

    assert body["git_commit"] == UNKNOWN
    assert body["build_time"] == UNKNOWN


# --- GET /api/v1/health --------------------------------------------------


async def test_health_returns_503_when_dependency_is_down(client):
    """Деградация отражается в статус-коде, а не только в теле ответа.

    Балансировщик и мониторинг смотрят на код, JSON они не разбирают.
    """
    response = await client.get("/api/v1/health")
    body = response.json()

    assert response.status_code == 503
    assert body["status"] == "degraded"
    assert body["components"]["postgres"]["status"] == "unavailable"
    assert body["components"]["postgres"]["error"]


async def test_health_returns_versions_and_latency_of_components(client, install_pool):
    """Основное требование: версия каждого third-party и время его ответа."""
    install_pool(FakePool())

    response = await client.get("/api/v1/health")
    body = response.json()

    assert response.status_code == 200
    assert body["status"] == "ok"
    assert body["version"] == read_pyproject_version()
    assert body["environment"] == "development"
    assert body["total_latency_ms"] >= 0

    postgres = body["components"]["postgres"]
    assert postgres["status"] == "healthy"
    assert postgres["version"] == "PostgreSQL 16.15"
    assert postgres["latency_ms"] >= 0
    assert postgres["error"] is None


async def test_health_reports_latency_even_for_failed_component(client):
    """Время ожидания отказа тоже попадает в отчёт."""
    body = (await client.get("/api/v1/health")).json()

    assert body["components"]["postgres"]["latency_ms"] >= 0


# Сквозные свойства


async def test_every_response_carries_request_id(client):
    """Каждый ответ помечен идентификатором для связывания логов."""
    response = await client.get("/healthz")

    assert response.headers["X-Request-ID"]
    assert len(response.headers["X-Request-ID"]) == 8


async def test_request_ids_are_unique_per_request(client):
    """Идентификаторы не переиспользуются между запросами."""
    first = await client.get("/healthz")
    second = await client.get("/healthz")

    assert first.headers["X-Request-ID"] != second.headers["X-Request-ID"]


async def test_openapi_version_matches_application_version(client):
    """Версия в OpenAPI-схеме берётся из того же источника, что и эндпоинт."""
    schema = (await client.get("/openapi.json")).json()

    assert schema["info"]["version"] == read_pyproject_version()
