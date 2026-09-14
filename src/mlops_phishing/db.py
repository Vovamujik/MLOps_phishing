"""Доступ к Postgres: пул соединений, его интерфейс и ошибки зависимости."""

import logging
from typing import Protocol

import asyncpg

from mlops_phishing.config import Settings

log = logging.getLogger(__name__)


class DatabasePool(Protocol):
    """Минимальный интерфейс пула, который нужен приложению.

    Сервисный слой зависит от этого протокола, а не от конкретного
    `asyncpg.Pool`: ему нужны ровно два метода. Выгода двойная — логика
    не привязана к драйверу, а в тестах можно подставить заглушку, и
    тайпчекер примет её без приведений типов.
    """

    async def fetchval(self, query: str) -> object:
        """Выполнить запрос и вернуть первое значение первой строки."""
        ...

    async def close(self) -> None:
        """Закрыть пул и все его соединения."""
        ...


DEPENDENCY_ERRORS = (OSError, asyncpg.PostgresError, TimeoutError)


def mask_dsn(dsn: str) -> str:
    """Убрать логин и пароль из строки подключения перед записью в лог."""
    return dsn.rsplit("@", 1)[-1]


async def create_pool(settings: Settings) -> asyncpg.Pool:
    """Создать пул соединений с Postgres.

    `min_size=0` — принципиальный момент: при старте не открывается ни одного
    соединения. Приложение поднимется, даже если база лежит, и `/api/v1/health`
    честно покажет её как недоступную. С `min_size>0` контейнер приложения
    падал бы в рестарт-луп из-за чужой аварии.
    """
    pool = await asyncpg.create_pool(
        settings.database_url,
        min_size=0,
        max_size=5,
        command_timeout=settings.health_check_timeout,
    )
    log.info("Пул соединений с Postgres создан: %s", mask_dsn(settings.database_url))
    return pool


async def close_pool(pool: asyncpg.Pool) -> None:
    """Закрыть пул и записать это в лог."""
    await pool.close()
    log.info("Пул соединений с Postgres закрыт")
