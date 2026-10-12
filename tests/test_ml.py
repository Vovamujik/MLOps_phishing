"""Тесты загрузки модели из MLflow Model Registry."""

import logging
from types import SimpleNamespace

import mlflow.sklearn
import pytest
from mlflow.exceptions import MlflowException

from mlops_phishing import ml
from mlops_phishing.config import Settings

from .conftest import DEAD_MLFLOW_URI, FakeClassifier, make_model


class FakeMlflowClient:
    """Реестр, где alias указывает на версию 7."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def get_model_version_by_alias(self, name: str, alias: str) -> SimpleNamespace:
        self.calls.append((name, alias))
        # Номер как int: так его отдаёт SQL-хранилище MLflow
        return SimpleNamespace(version=7, run_id="run-7")


@pytest.fixture
def registry(monkeypatch):
    """Подменить клиент реестра и загрузку артефакта, запоминая вызовы."""
    fake = SimpleNamespace(
        client=FakeMlflowClient(), loaded_uris=[], dst_paths=[], tracking_uris=[]
    )

    def fake_load(uri, dst_path=None):
        fake.loaded_uris.append(uri)
        fake.dst_paths.append(dst_path)
        return FakeClassifier()

    monkeypatch.setattr(ml, "MlflowClient", lambda: fake.client)
    monkeypatch.setattr(ml.mlflow, "set_tracking_uri", fake.tracking_uris.append)
    monkeypatch.setattr(mlflow.sklearn, "load_model", fake_load)
    return fake


def test_load_model_resolves_alias_once_and_loads_that_exact_version(registry):
    """Грузится models:/name/7, а не @champion повторно: alias мог бы переехать."""
    model = ml.load_model(Settings(mlflow_tracking_uri="http://mlflow:5000"))

    assert registry.tracking_uris == ["http://mlflow:5000"]
    assert registry.client.calls == [("phishing-url-clf", "champion")]
    assert registry.loaded_uris == ["models:/phishing-url-clf/7"]
    # Скачивание во временную папку, которую удаляем: иначе каждый старт копит копию модели
    assert registry.dst_paths[0] is not None
    assert (model.name, model.version, model.alias, model.run_id) == (
        "phishing-url-clf",
        "7",
        "champion",
        "run-7",
    )


def test_alias_comes_from_settings(registry):
    """Сменить обслуживаемую версию можно настройкой, без правки кода."""
    ml.load_model(Settings(model_alias="challenger"))

    assert registry.client.calls == [("phishing-url-clf", "challenger")]


async def test_load_model_or_none_logs_loaded_version(monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger="mlops_phishing.ml")
    monkeypatch.setattr(ml, "load_model", lambda settings: make_model(version="7"))

    model = await ml.load_model_or_none(Settings())

    assert model is not None
    assert model.version == "7"
    assert "phishing-url-clf v7" in caplog.text


async def test_missing_alias_does_not_crash_startup(monkeypatch, caplog):
    caplog.set_level(logging.ERROR, logger="mlops_phishing.ml")

    def missing(settings):
        raise MlflowException("Registered model alias champion not found.")

    monkeypatch.setattr(ml, "load_model", missing)

    assert await ml.load_model_or_none(Settings()) is None
    assert "не загружена" in caplog.text


async def test_broken_artifact_does_not_crash_startup(monkeypatch, caplog):
    """Битый или несовместимый файл модели — тоже старт без модели, а не рестарт-луп."""
    caplog.set_level(logging.ERROR, logger="mlops_phishing.ml")

    def broken(settings):
        raise TypeError("Untrusted types found in the file")

    monkeypatch.setattr(ml, "load_model", broken)

    assert await ml.load_model_or_none(Settings()) is None
    assert "не десериализовалась" in caplog.text


async def test_unreachable_mlflow_does_not_crash_startup(monkeypatch, caplog):
    """Настоящий клиент MLflow на закрытом порту: отказ за секунду, приложение без модели."""
    caplog.set_level(logging.ERROR, logger="mlops_phishing.ml")
    monkeypatch.setenv("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "0")
    # set_tracking_uri пишет адрес и в окружение: так monkeypatch его вернёт
    monkeypatch.setenv("MLFLOW_TRACKING_URI", DEAD_MLFLOW_URI)

    assert await ml.load_model_or_none(Settings(mlflow_tracking_uri=DEAD_MLFLOW_URI)) is None
    assert "MlflowException" in caplog.text


def test_phishing_probability_is_the_positive_class_column():
    model = make_model(FakeClassifier(phishing_probability=0.8))

    assert model.phishing_probability("https://example.com") == pytest.approx(0.8)
