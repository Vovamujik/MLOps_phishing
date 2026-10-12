"""Общая настройка MLflow для скриптов обучения и EDA."""

import os

import mlflow

DEFAULT_TRACKING_URI = "http://localhost:5050"


def setup_mlflow(experiment: str) -> None:
    """Подключается к MLflow из docker compose и выбирает эксперимент.

    Переменные окружения, если заданы, важнее значений по умолчанию.
    """
    os.environ.setdefault("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI)
    # Иначе клиент скачивает артефакты напрямую с s3:9000, а этот адрес виден только внутри compose
    os.environ.setdefault("MLFLOW_ENABLE_PROXY_MULTIPART_DOWNLOAD", "false")
    mlflow.set_experiment(experiment)
