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
from mlops_phishing.config import get_settings
from mlops_phishing.metadata import get_version

log = logging.getLogger(__name__)

REQUEST_ID_HEADER = "X-Request-ID"


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
        try:
            response = await call_next(request)
            elapsed_ms = (time.perf_counter() - started_at) * 1000
            log.info(
                "%s %s -> %d (%.1f ms)",
                request.method,
                request.url.path,
                response.status_code,
                elapsed_ms,
            )
        finally:
            logging_config.REQUEST_ID.reset(token)

        response.headers[REQUEST_ID_HEADER] = request_id
        return response

    app.include_router(root_router)
    return app


app = create_app()
