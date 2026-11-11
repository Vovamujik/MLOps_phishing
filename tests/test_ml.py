"""Тесты загрузки модели из MLflow Model Registry."""

from types import SimpleNamespace

import mlflow.sklearn as mlflow_sklearn
import pytest

from mlops_phishing import ml
from mlops_phishing.config import Settings

from .conftest import FakeClassifier, make_model


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
    monkeypatch.setattr(mlflow_sklearn, "load_model", fake_load)
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


def test_phishing_probability_is_the_positive_class_column():
    classifier = FakeClassifier(phishing_probability=0.8)
    model = make_model(classifier)

    assert model.phishing_probability("https://example.com") == pytest.approx(0.8)
    # Модель обучена на колонке URL (training/train.py), отдаём ей ровно её
    assert list(classifier.frames[0].columns) == ["URL"]
    assert classifier.frames[0]["URL"].tolist() == ["https://example.com"]
