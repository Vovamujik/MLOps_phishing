"""Запуск настоящих сервисов для интеграционных тестов.

Образы те же, что в docker-compose.yml: тест проверяет то, что поедет в стек.
Без Docker тесты пропускаются, а не падают.
"""

import asyncio
import time
from collections.abc import Iterator
from contextlib import contextmanager

import asyncpg
import pytest

POSTGRES_IMAGE = "postgres:16-alpine"


def docker_available() -> bool:
    try:
        from testcontainers.core.docker_client import DockerClient

        DockerClient().client.ping()
    except Exception:  # noqa: BLE001 — любая ошибка значит «Docker недоступен»
        return False
    return True


@contextmanager
def start_postgres() -> Iterator[str]:
    """Postgres в контейнере. Возвращает DSN, когда база принимает подключения."""
    if not docker_available():
        pytest.skip("Docker недоступен")
    from testcontainers.core.container import DockerContainer

    container = (
        DockerContainer(POSTGRES_IMAGE)
        .with_env("POSTGRES_USER", "test")
        .with_env("POSTGRES_PASSWORD", "test")
        .with_env("POSTGRES_DB", "test")
        .with_exposed_ports(5432)
    )
    with container:
        host = container.get_container_host_ip()
        port = container.get_exposed_port(5432)
        dsn = f"postgresql://test:test@{host}:{port}/test"
        _wait_for_postgres(dsn)
        yield dsn


def _wait_for_postgres(dsn: str, timeout: float = 60) -> None:
    async def probe() -> None:
        connection = await asyncpg.connect(dsn, timeout=2)
        await connection.close()

    deadline = time.monotonic() + timeout
    while True:
        try:
            asyncio.run(probe())
        except (OSError, asyncpg.PostgresError, TimeoutError):
            if time.monotonic() > deadline:
                raise
            time.sleep(0.5)
        else:
            return
