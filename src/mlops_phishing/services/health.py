"""Сбор end-to-end health-отчёта.

Разделение ответственности:

- `read_*_version` — знает только, как спросить у конкретного компонента его
  версию. Ни таймаутов, ни обработки ошибок, ни замера времени.
- `run_check` — универсальная обёртка: накладывает таймаут, замеряет время
  ответа и превращает отказ в статус `unavailable`.
- `build_report` — запускает все проверки параллельно и сводит результат.

Благодаря этому добавление новой зависимости (Redis, S3) — это одна короткая
корутина и одна строка в `checks`, без копирования логики таймаутов.
"""

import asyncio
import logging
from collections.abc import Awaitable
from time import perf_counter

from mlops_phishing.config import Settings
from mlops_phishing.db import DEPENDENCY_ERRORS, DatabasePool
from mlops_phishing.metadata import get_version
from mlops_phishing.schemas import ComponentHealth, ComponentStatus, HealthReport, ReportStatus

log = logging.getLogger(__name__)

POSTGRES_VERSION_QUERY = "SELECT version()"


def elapsed_ms(started_at: float) -> float:
    """Сколько миллисекунд прошло с отметки времени."""
    return round((perf_counter() - started_at) * 1000, 2)


def parse_postgres_version(raw: str) -> str:
    """Выделить короткую версию из ответа `SELECT version()`."""
    parts = raw.split()
    return " ".join(parts[:2]) if len(parts) >= 2 else raw


async def read_postgres_version(pool: DatabasePool) -> str:
    """Спросить у Postgres его версию."""
    raw = await pool.fetchval(POSTGRES_VERSION_QUERY)
    return parse_postgres_version(str(raw))


async def run_check(probe: Awaitable[str], limit_seconds: float) -> ComponentHealth:
    """Выполнить проверку под таймаутом, замерив время ответа.

    Недоступный компонент попадает в отчёт со статусом `unavailable` и текстом
    причины для дежурного. Время ответа возвращается и при отказе: сколько
    ждали до него — тоже диагностика.
    """
    started_at = perf_counter()
    try:
        async with asyncio.timeout(limit_seconds):
            version = await probe
    except DEPENDENCY_ERRORS as exc:
        log.warning("Компонент недоступен: %s: %s", type(exc).__name__, exc)
        return ComponentHealth(
            status=ComponentStatus.unavailable,
            latency_ms=elapsed_ms(started_at),
            error=f"{type(exc).__name__}: {exc}",
        )

    return ComponentHealth(
        status=ComponentStatus.healthy,
        version=version,
        latency_ms=elapsed_ms(started_at),
    )


async def build_report(pool: DatabasePool, settings: Settings) -> HealthReport:
    """Собрать отчёт по всем third-party компонентам.

    `asyncio.gather` запускает проверки параллельно: добавление пятой
    зависимости не увеличит время ответа эндпоинта. Последовательный вариант
    складывал бы таймауты, и при двух лежащих компонентах мониторинг сам
    отвалился бы по таймауту, не дождавшись ответа.
    """
    checks: dict[str, Awaitable[str]] = {
        "postgres": read_postgres_version(pool),
    }

    started_at = perf_counter()
    results = await asyncio.gather(
        *(run_check(probe, settings.health_check_timeout) for probe in checks.values())
    )
    total_latency_ms = elapsed_ms(started_at)

    components = dict(zip(checks, results, strict=True))
    all_healthy = all(item.status is ComponentStatus.healthy for item in components.values())

    return HealthReport(
        status=ReportStatus.ok if all_healthy else ReportStatus.degraded,
        version=get_version(),
        environment=settings.environment,
        total_latency_ms=total_latency_ms,
        components=components,
    )
