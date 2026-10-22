"""Запуск настоящих сервисов для интеграционных тестов.

Образы те же, что в docker-compose.yml: тест проверяет то, что поедет в стек.
Без Docker локально тесты пропускаются, а в CI падают: там Docker есть всегда,
и пропуск скрыл бы, что тесты не шли.
"""

import asyncio
import os
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import asyncpg
import pytest

POSTGRES_IMAGE = "postgres:16-alpine"
RABBITMQ_IMAGE = "rabbitmq:4.3.6-management"
AMQP_PORT = 5672
MANAGEMENT_PORT = 15672


def docker_available() -> bool:
    try:
        from testcontainers.core.docker_client import DockerClient

        DockerClient().client.ping()
    except Exception:  # noqa: BLE001 — любая ошибка значит «Docker недоступен»
        return False
    return True


def require_docker() -> None:
    if not docker_available():
        # GitHub Actions выставляет CI=true
        (pytest.fail if os.environ.get("CI") else pytest.skip)("Docker недоступен")


@pytest.fixture(scope="session")
def postgres_dsn() -> Iterator[str]:
    """Один Postgres на все интеграционные тесты."""
    with start_postgres() as dsn:
        yield dsn


@contextmanager
def start_postgres() -> Iterator[str]:
    """Postgres в контейнере. Возвращает DSN, когда база принимает подключения."""
    require_docker()
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


@dataclass(frozen=True)
class Rabbit:
    """Запущенный RabbitMQ: адреса AMQP и HTTP API управления, сам контейнер."""

    amqp_url: str
    management_url: str
    container: Any

    def ctl(self, *args: str) -> None:
        """rabbitmqctl внутри контейнера."""
        result = self.container.exec(["rabbitmqctl", *args])
        assert result.exit_code == 0, result.output


@contextmanager
def start_rabbitmq() -> Iterator[Rabbit]:
    """RabbitMQ в контейнере; готов, когда принимает AMQP-соединения."""
    require_docker()
    from testcontainers.community.rabbitmq import RabbitMqContainer

    container = (
        RabbitMqContainer(RABBITMQ_IMAGE, username="app", password="app")  # noqa: S106 — тестовый
        .with_exposed_ports(MANAGEMENT_PORT)
        # Docker Desktop 29 не даёт Erlang прочитать свой cookie — тот же обход, что в compose
        .with_env("RABBITMQ_SERVER_ADDITIONAL_ERL_ARGS", "-setcookie integration")
        .with_env("RABBITMQ_CTL_ERL_ARGS", "-setcookie integration")
    )
    with container:
        host = container.get_container_host_ip()
        yield Rabbit(
            amqp_url=f"amqp://app:app@{host}:{container.get_exposed_port(AMQP_PORT)}/",
            management_url=f"http://{host}:{container.get_exposed_port(MANAGEMENT_PORT)}",
            container=container,
        )
