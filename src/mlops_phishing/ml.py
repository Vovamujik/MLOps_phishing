"""Модель из MLflow Model Registry: загрузка при старте воркера и предсказание.

Воркер знает только имя модели и alias. Какая версия стоит за alias, решает
реестр: переставили alias, перезапустили воркер, и обслуживается другая
версия без правки кода.
"""

import os
import tempfile
from dataclasses import dataclass
from typing import Protocol, cast

import mlflow
import mlflow.sklearn as mlflow_sklearn
import numpy as np
import pandas as pd
from mlflow import MlflowClient

from mlops_phishing.config import Settings

INPUT_COLUMN = "URL"


class Classifier(Protocol):
    """Всё, что воркеру нужно от модели: вероятности классов."""

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        """Вернуть массив (n, 2): вероятности легитимного сайта и фишинга."""
        ...


@dataclass(frozen=True)
class LoadedModel:
    """Загруженная модель и её происхождение в реестре."""

    name: str
    version: str
    alias: str
    # У версии, зарегистрированной не из рана, run_id нет
    run_id: str | None
    classifier: Classifier

    def phishing_probability(self, url: str) -> float:
        """Вероятность фишинга для одного URL. Блокирует: это CPU-работа sklearn."""
        frame = pd.DataFrame({INPUT_COLUMN: [url]})
        return float(self.classifier.predict_proba(frame)[0, 1])


def load_model(settings: Settings) -> LoadedModel:
    """Загрузить версию, на которую указывает alias. Блокирует: сеть и диск.

    Alias разрешается в номер версии один раз, и грузится именно эта версия.
    Если загрузить `@alias` вторым запросом, alias может переехать между ними,
    и воркер записал бы в результаты не ту версию, что загрузил.
    """
    configure_mlflow_client(settings)
    version = MlflowClient().get_model_version_by_alias(settings.model_name, settings.model_alias)
    # Через REST номер версии — строка, из SQL-хранилища — int
    number = str(version.version)
    # Файлы модели нужны только на время десериализации, дальше она живёт в памяти
    with tempfile.TemporaryDirectory() as tmp:
        # У MLflow нет аннотаций, и pyright видит в ответе None. Его не бывает:
        # неизвестный формат модели MLflow отвергает раньше, исключением
        classifier = cast(
            Classifier,
            mlflow_sklearn.load_model(f"models:/{settings.model_name}/{number}", dst_path=tmp),
        )
    return LoadedModel(
        name=settings.model_name,
        version=number,
        alias=settings.model_alias,
        run_id=version.run_id,
        classifier=classifier,
    )


def configure_mlflow_client(settings: Settings) -> None:
    """Настроить клиент MLflow одинаково в compose и при локальном запуске.

    Переменные окружения, если заданы, важнее значений по умолчанию.
    """
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    # Лежащий MLflow не держит старт минутами: по умолчанию 7 попыток, около 4 минут
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "2")
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "10")
    # Файлы модели идут через сервер MLflow, а не напрямую из s3, который виден только в compose
    os.environ.setdefault("MLFLOW_ENABLE_PROXY_MULTIPART_DOWNLOAD", "false")
