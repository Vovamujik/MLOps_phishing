"""Общие фикстуры тестов."""

import asyncio
import logging
import threading
import tomllib
from collections.abc import AsyncIterator, Callable, Iterator
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from asgi_lifespan import LifespanManager
from httpx import ASGITransport, AsyncClient

from mlops_phishing.app import app
from mlops_phishing.config import Settings, get_settings
from mlops_phishing.metadata import get_version
from mlops_phishing.ml import LoadedModel

PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"


def read_pyproject_version() -> str:
    """Прочитать версию напрямую из pyproject.toml.

    Тесты сверяют ответы приложения с этим значением, поэтому читают файл
    сами, а не через тот же код, который проверяют.
    """
    with PYPROJECT.open("rb") as file:
        return tomllib.load(file)["project"]["version"]


# Заведомо закрытый порт попытка подключения падает сразу
DEAD_DATABASE_URL = "postgresql://postgres:postgres@127.0.0.1:59999/mlops_phishing"
DEAD_MLFLOW_URI = "http://127.0.0.1:59998"

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


class FakeClassifier:
    """Заглушка sklearn-модели: отдаёт заданную вероятность фишинга и запоминает вызовы."""

    def __init__(self, phishing_probability: float = 0.9) -> None:
        self.phishing_probability = phishing_probability
        self.frames: list[pd.DataFrame] = []
        self.thread_ids: list[int] = []

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        """Вернуть [легит, фишинг] для каждой строки и запомнить вход и поток."""
        self.frames.append(x)
        self.thread_ids.append(threading.get_ident())
        p = self.phishing_probability
        return np.array([[1 - p, p]] * len(x))


def make_model(classifier: FakeClassifier | None = None, version: str = "3") -> LoadedModel:
    """Модель, как будто загруженная из реестра."""
    return LoadedModel(
        name="phishing-url-clf",
        version=version,
        alias="champion",
        run_id="run-123",
        classifier=classifier or FakeClassifier(),
    )


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
    monkeypatch.setenv("MLFLOW_TRACKING_URI", DEAD_MLFLOW_URI)
    monkeypatch.setenv("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "0")
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


@pytest.fixture
def install_model(monkeypatch: pytest.MonkeyPatch) -> Callable[[LoadedModel], LoadedModel]:
    """Подменить загруженную модель в состоянии приложения."""

    def _install(model: LoadedModel) -> LoadedModel:
        monkeypatch.setattr(app.state, "model", model, raising=False)
        return model

    return _install
