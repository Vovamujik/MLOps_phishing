"""Общие фикстуры тестов."""

import asyncio
import logging
import tomllib
from collections.abc import AsyncIterator, Callable, Iterator
from pathlib import Path

import pytest
from asgi_lifespan import LifespanManager
from httpx import ASGITransport, AsyncClient

from mlops_phishing.app import app
from mlops_phishing.config import Settings, get_settings
from mlops_phishing.metadata import get_version

PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"


def read_pyproject_version() -> str:
    """Прочитать версию напрямую из pyproject.toml.

    Тесты сверяют ответы приложения с этим значением, поэтому читают файл
    сами, а не через тот же код, который проверяют.
    """
    with PYPROJECT.open("rb") as file:
        return tomllib.load(file)["project"]["version"]


# Заведомо закрытый порт: попытка подключения падает сразу, а не ждёт таймаут
DEAD_DATABASE_URL = "postgresql://postgres:postgres@127.0.0.1:59999/mlops_phishing"

POSTGRES_VERSION_RAW = "PostgreSQL 16.15 (Debian 16.15-1) on aarch64-unknown-linux-gnu, gcc"


class FakePool:
    """Заглушка пула asyncpg: отвечает заданной версией, задержкой или ошибкой."""

    def __init__(
        self,
        *,
        version: str = POSTGRES_VERSION_RAW,
        delay: float = 0.0,
        error: BaseException | None = None,
    ) -> None:
        self.version = version
        self.delay = delay
        self.error = error
        self.queries: list[str] = []
        self.closed = False

    async def fetchval(self, query: str) -> str:
        """Вернуть версию, предварительно изобразив задержку или отказ."""
        self.queries.append(query)
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.error is not None:
            raise self.error
        return self.version

    async def close(self) -> None:
        """Повторить контракт настоящего пула: его закрывает shutdown-фаза lifespan."""
        self.closed = True


@pytest.fixture(autouse=True)
def isolate_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Изолировать тесты от локального окружения разработчика.

    1. `.env` в корне репозитория есть у разработчика и отсутствует в CI —
       без отключения тесты вели бы себя по-разному.
    2. Кэши `get_settings` и `get_version` иначе протекают между тестами,
       и порядок выполнения начинает влиять на результат.
    """
    monkeypatch.setitem(Settings.model_config, "env_file", None)
    get_settings.cache_clear()
    get_version.cache_clear()


@pytest.fixture(autouse=True)
def restore_root_logger() -> Iterator[None]:
    """Вернуть корневой логгер в исходное состояние после теста.

    `setup_logging` переписывает обработчики корневого логгера. Без отката
    он ломал бы захват вывода pytest в последующих тестах.
    """
    root = logging.getLogger()
    handlers = root.handlers[:]
    level = root.level
    yield
    root.handlers = handlers
    root.setLevel(level)


@pytest.fixture
async def client(monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[AsyncClient]:
    """HTTP-клиент приложения с поднятым lifespan.

    `LifespanManager` выполняет startup и shutdown, иначе `app.state.db_pool`
    не существует и health-эндпоинт падал бы на пустом состоянии.
    `ASGITransport` вызывает приложение напрямую, без сокета и порта.
    """
    monkeypatch.setenv("DATABASE_URL", DEAD_DATABASE_URL)
    get_settings.cache_clear()

    async with LifespanManager(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as http_client:
            yield http_client


@pytest.fixture
def install_pool(monkeypatch: pytest.MonkeyPatch) -> Callable[[FakePool], FakePool]:
    """Подменить пул БД в состоянии приложения на заглушку."""

    def _install(pool: FakePool) -> FakePool:
        monkeypatch.setattr(app.state, "db_pool", pool, raising=False)
        return pool

    return _install
