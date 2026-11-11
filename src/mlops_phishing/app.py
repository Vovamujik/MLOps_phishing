"""Точка сборки приложения.

Запуск локально: `uv run uvicorn mlops_phishing.app:app --reload`
"""

import asyncio
import logging
import time
import uuid
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, Request
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from starlette.responses import Response

from mlops_phishing import db, instruments, logging_config, tasks_repo, telemetry
from mlops_phishing.api.router import API_V1_PREFIX, root_router
from mlops_phishing.api.system import LIVENESS_PATH
from mlops_phishing.config import get_settings
from mlops_phishing.metadata import get_version
from mlops_phishing.publisher import RabbitTaskPublisher

log = logging.getLogger(__name__)

REQUEST_ID_HEADER = "X-Request-ID"

# Пробы Docker и мониторинга: дёргаются постоянно, на INFO засоряли бы лог
QUIET_PATHS = frozenset({LIVENESS_PATH, f"{API_V1_PREFIX}/health"})
# Пробы и в трейсы не пишем: тысячи одинаковых трейсов без пользы (регулярки по URL)
TRACE_EXCLUDED_URLS = ",".join(path.lstrip("/") + "$" for path in sorted(QUIET_PATHS))


def access_log_level(path: str, status_code: int, shedding: bool = False) -> int:
    """Выбрать уровень записи о запросе.

    Уровень зависит от ответа, чтобы проблемные запросы было видно сразу.
    Тихие пути понижаются до DEBUG только при успехе: упавший healthcheck
    должен быть так же заметен, как любая другая ошибка. Плановый сброс
    нагрузки (503 с Retry-After) — штатная работа под перегрузкой, а не авария.
    """
    if status_code >= 500:
        return logging.WARNING if shedding else logging.ERROR
    if status_code >= 400:
        return logging.WARNING
    if path in QUIET_PATHS:
        return logging.DEBUG
    return logging.INFO


def access_fields(request: Request, started_at: float, status_code: int) -> dict[str, object]:
    """Поля строки access-лога: код, длительность и task_id, если запрос создал задачу."""
    fields: dict[str, object] = {
        "status_code": status_code,
        "duration_ms": round((time.perf_counter() - started_at) * 1000, 1),
    }
    task_id = getattr(request.state, "task_id", None)
    if task_id is not None:
        fields["task_id"] = str(task_id)
    return fields


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    """Жизненный цикл: пул БД и издатель задач при старте, закрыть их при остановке.

    Ресурсы создаются здесь, а не на уровне модуля: при импорте ещё нет
    работающего event loop, а тесты должны уметь поднимать и гасить
    приложение многократно. Модели в API нет: её грузит воркер.
    """
    settings = get_settings()
    app.state.settings = settings
    app.state.db_pool = await db.create_pool(settings)
    app.state.publisher = RabbitTaskPublisher(settings)
    try:
        async with asyncio.timeout(settings.health_check_timeout):
            await tasks_repo.ensure_schema(app.state.db_pool)
    except db.DEPENDENCY_ERRORS as exc:
        # База лежит — API всё равно стартует (как с min_size=0), таблицу создаст воркер
        log.warning("Таблица задач не проверена: %s: %s", type(exc).__name__, exc)

    log.info(
        "Приложение %s v%s запущено (environment=%s, debug=%s)",
        settings.app_name,
        get_version(),
        settings.environment,
        settings.debug,
    )
    try:
        yield
    finally:
        # uvicorn уже перестал принимать запросы и дождался текущих
        log.info("Остановка API: закрываем соединения с брокером и базой")
        await app.state.publisher.close()
        await db.close_pool(app.state.db_pool)
        log.info("API остановлен")
        # Последним: спаны остановки тоже должны уйти в Collector
        app.state.telemetry.shutdown()


def create_app() -> FastAPI:
    """Собрать и настроить приложение FastAPI."""
    settings = get_settings()
    # При импорте, а не в lifespan: строки uvicorn о старте сервера идут раньше lifespan
    logging_config.setup_logging(settings.log_level, "api", settings.environment)
    tracing = telemetry.setup_telemetry("api", instrument_asyncpg=True)

    app = FastAPI(
        title=settings.app_name,
        version=get_version(),
        description="MLOps-сервис обнаружения фишинговых сайтов по URL.",
        lifespan=lifespan,
    )

    @app.middleware("http")
    async def attach_request_id(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        """Присвоить запросу идентификатор и залогировать его результат.

        request_id попадает в contextvars и оттуда во все записи запроса. task_id
        обработчик кладёт в request.state: его contextvars в middleware не видны.
        """
        request_id = uuid.uuid4().hex[:8]
        started_at = time.perf_counter()
        fields: dict[str, object] = {
            "method": request.method,
            "path": request.url.path,
            "client": request.client.host if request.client else "-",
        }
        with structlog.contextvars.bound_contextvars(request_id=request_id):
            try:
                response = await call_next(request)
            except Exception as exc:
                log.error(
                    "%s %s -> 500",
                    request.method,
                    request.url.path,
                    extra=fields | access_fields(request, started_at, 500),
                    exc_info=exc,
                )
                raise
            log.log(
                access_log_level(
                    request.url.path, response.status_code, "Retry-After" in response.headers
                ),
                "%s %s -> %d",
                request.method,
                request.url.path,
                response.status_code,
                extra=fields | access_fields(request, started_at, response.status_code),
            )

        response.headers[REQUEST_ID_HEADER] = request_id
        return response

    app.include_router(root_router)
    app.state.telemetry = tracing
    instruments.observe_queue_limit(settings.tasks_queue_max_length)
    # Спан на запрос; внутренние спаны ASGI send/receive — шум, их не пишем
    FastAPIInstrumentor.instrument_app(
        app, excluded_urls=TRACE_EXCLUDED_URLS, exclude_spans=["receive", "send"]
    )
    return app


app = create_app()
