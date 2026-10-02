"""Точка сборки приложения.

Запуск локально: `uv run uvicorn mlops_phishing.app:app --reload`
"""

import logging
import time
import uuid
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from starlette.responses import Response

from mlops_phishing import db, logging_config
from mlops_phishing.api.router import root_router
from mlops_phishing.api.system import LIVENESS_PATH
from mlops_phishing.config import get_settings
from mlops_phishing.metadata import get_version

log = logging.getLogger(__name__)

REQUEST_ID_HEADER = "X-Request-ID"

QUIET_PATHS = frozenset({LIVENESS_PATH})


def access_log_level(path: str, status_code: int) -> int:
    """Выбрать уровень записи о запросе.

    Уровень зависит от ответа, чтобы проблемные запросы было видно сразу.
    Тихие пути понижаются до DEBUG только при успехе: упавший healthcheck
    должен быть так же заметен, как любая другая ошибка.
    """
    if status_code >= 500:
        return logging.ERROR
    if status_code >= 400:
        return logging.WARNING
    if path in QUIET_PATHS:
        return logging.DEBUG
    return logging.INFO


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
    """Жизненный цикл: поднять пул БД при старте, закрыть при остановке.

    Ресурсы создаются здесь, а не на уровне модуля: при импорте ещё нет
    работающего event loop, а тесты должны уметь поднимать и гасить
    приложение многократно.
    """
    settings = get_settings()
    logging_config.setup_logging(settings.log_level)

    app.state.settings = settings
    app.state.db_pool = await db.create_pool(settings)

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
        await db.close_pool(app.state.db_pool)
        log.info("Приложение остановлено")


def create_app() -> FastAPI:
    """Собрать и настроить приложение FastAPI."""
    settings = get_settings()

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
        """Присвоить запросу идентификатор и залогировать его результат."""
        request_id = uuid.uuid4().hex[:8]
        token = logging_config.REQUEST_ID.set(request_id)
        started_at = time.perf_counter()
        client = request.client.host if request.client else "-"
        try:
            response = await call_next(request)
        except Exception as exc:
            log.error(
                "%s %s -> 500 (%.1f ms) client=%s exception=%s",
                request.method,
                request.url.path,
                (time.perf_counter() - started_at) * 1000,
                client,
                type(exc).__name__,
            )
            raise
        else:
            log.log(
                access_log_level(request.url.path, response.status_code),
                "%s %s -> %d (%.1f ms) client=%s",
                request.method,
                request.url.path,
                response.status_code,
                (time.perf_counter() - started_at) * 1000,
                client,
            )
        finally:
            logging_config.REQUEST_ID.reset(token)

        response.headers[REQUEST_ID_HEADER] = request_id
        return response

    app.include_router(root_router)
    return app


app = create_app()
