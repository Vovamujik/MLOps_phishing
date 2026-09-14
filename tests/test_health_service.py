"""Тесты логики health-отчёта."""

import asyncio

from mlops_phishing.config import Settings
from mlops_phishing.schemas import ComponentStatus, ReportStatus
from mlops_phishing.services import health as health_service

from .conftest import POSTGRES_VERSION_RAW, FakePool


def test_parse_version_extracts_product_and_number():
    """Из длинной строки Postgres берутся только продукт и номер версии."""
    assert health_service.parse_postgres_version(POSTGRES_VERSION_RAW) == "PostgreSQL 16.15"


def test_parse_version_returns_short_input_unchanged():
    """Неожиданно короткий ответ возвращается как есть, а не роняет отчёт."""
    assert health_service.parse_postgres_version("PostgreSQL") == "PostgreSQL"


async def test_read_postgres_version_queries_the_database():
    """Проверка действительно выполняет запрос к базе, а не выдумывает ответ."""
    pool = FakePool()

    version = await health_service.read_postgres_version(pool)

    assert version == "PostgreSQL 16.15"
    assert pool.queries == [health_service.POSTGRES_VERSION_QUERY]


async def test_run_check_reports_version_and_latency():
    """Здоровый компонент отдаёт версию и измеренное время ответа."""
    component = await health_service.run_check(
        health_service.read_postgres_version(FakePool()), limit_seconds=1.0
    )

    assert component.status is ComponentStatus.healthy
    assert component.version == "PostgreSQL 16.15"
    assert component.latency_ms >= 0
    assert component.error is None


async def test_run_check_reports_unavailable_on_connection_error():
    """Отказ соединения становится статусом, а не исключением наружу."""
    pool = FakePool(error=OSError("connection refused"))

    component = await health_service.run_check(
        health_service.read_postgres_version(pool), limit_seconds=1.0
    )

    assert component.status is ComponentStatus.unavailable
    assert component.version is None
    assert "OSError" in (component.error or "")


async def test_run_check_gives_up_after_limit():
    """Зависший компонент обрывается по таймауту, а не держит эндпоинт."""
    pool = FakePool(delay=5.0)

    component = await health_service.run_check(
        health_service.read_postgres_version(pool), limit_seconds=0.01
    )

    assert component.status is ComponentStatus.unavailable
    assert "TimeoutError" in (component.error or "")
    # Ждали ровно отведённый лимит, а не пять секунд.
    assert component.latency_ms < 1000


async def test_run_check_measures_latency_even_on_failure():
    """Сколько именно ждали до отказа — тоже диагностика."""
    pool = FakePool(delay=0.02, error=OSError("boom"))

    component = await health_service.run_check(
        health_service.read_postgres_version(pool), limit_seconds=1.0
    )

    assert component.status is ComponentStatus.unavailable
    assert component.latency_ms >= 20


async def test_build_report_is_ok_when_every_component_is_healthy():
    """Все зависимости здоровы — общий статус ok, версии на месте."""
    report = await health_service.build_report(FakePool(), Settings())

    assert report.status is ReportStatus.ok
    assert report.environment == "development"
    assert report.components["postgres"].version == "PostgreSQL 16.15"
    assert report.total_latency_ms >= 0


async def test_build_report_degrades_when_component_is_down():
    """Одна нездоровая зависимость роняет общий статус до degraded."""
    report = await health_service.build_report(FakePool(error=OSError("down")), Settings())

    assert report.status is ReportStatus.degraded
    assert report.components["postgres"].status is ComponentStatus.unavailable


async def test_checks_run_concurrently_not_sequentially():
    """Проверки идут параллельно: общее время — максимум, а не сумма.

    Последовательный вариант складывал бы таймауты, и при нескольких лежащих
    компонентах мониторинг сам отвалился бы, не дождавшись ответа.
    """
    delay = 0.05
    probes = [health_service.read_postgres_version(FakePool(delay=delay)) for _ in range(4)]

    started_at = asyncio.get_running_loop().time()
    await asyncio.gather(*(health_service.run_check(probe, 1.0) for probe in probes))
    elapsed = asyncio.get_running_loop().time() - started_at

    assert elapsed < delay * 2, "проверки выполнялись последовательно"
