"""Обучение вариантов модели.

Запуск: uv run python -m training.train baseline
"""

import argparse
import time
from collections.abc import Callable
from dataclasses import dataclass

import mlflow
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from training.data import TARGET, Splits, load_dataset, log_splits, split_by_domain
from training.evaluation import THRESHOLD, compute_metrics, log_diagnostics
from training.tracking import setup_mlflow

EXPERIMENT = "phishing-url-clf"


def tfidf_logreg(ngram_min: int, ngram_max: int, min_df: int, c: float) -> Pipeline:
    """TF-IDF по символьным n-граммам URL, затем логистическая регрессия.

    c — обратная сила регуляризации (C в sklearn).
    """
    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(ngram_min, ngram_max),
        min_df=min_df,
        sublinear_tf=True,
        dtype=np.float32,
    )
    return Pipeline(
        [
            ("features", ColumnTransformer([("url", vectorizer, "URL")])),
            ("classifier", LogisticRegression(C=c, max_iter=1000)),
        ]
    )


@dataclass(frozen=True)
class Variant:
    build: Callable[..., Pipeline]
    params: dict[str, object]
    # Колонки, которые модель получает на вход. Для сервиса это только URL
    columns: tuple[str, ...] = ("URL",)


VARIANTS: dict[str, Variant] = {
    "baseline": Variant(tfidf_logreg, {"ngram_min": 3, "ngram_max": 5, "min_df": 10, "c": 1.0}),
}


def run_variant(name: str, variant: Variant, splits: Splits) -> None:
    columns = list(variant.columns)
    model = variant.build(**variant.params)

    with mlflow.start_run(run_name=name):
        mlflow.set_tag("variant", name)
        log_splits(splits)
        mlflow.log_params(
            variant.params | {"input_columns": ",".join(columns), "threshold": THRESHOLD}
        )

        started = time.perf_counter()
        model.fit(splits.train[columns], splits.train[TARGET])
        mlflow.log_metric("train_seconds", time.perf_counter() - started)

        val_proba = _predict(model, splits.val, columns)
        mlflow.log_metrics(_prefixed("val", compute_metrics(splits.val[TARGET], val_proba)))

        started = time.perf_counter()
        test_proba = _predict(model, splits.test, columns)
        mlflow.log_metric(
            "predict_ms_per_1k", (time.perf_counter() - started) / len(splits.test) * 1e6
        )
        mlflow.log_metrics(_prefixed("test", compute_metrics(splits.test[TARGET], test_proba)))

        log_diagnostics(model, splits.test, test_proba)
        # После метрик: MLflow копирует метрики рана в карточку модели
        log_model(model, splits.val[columns])


def _predict(model: Pipeline, part: pd.DataFrame, columns: list[str]) -> np.ndarray:
    return model.predict_proba(part[columns])[:, 1]


def _prefixed(prefix: str, metrics: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def log_model(model: Pipeline, sample: pd.DataFrame) -> None:
    sample = sample.head(100)
    signature = infer_signature(sample, model.predict_proba(sample))
    mlflow.sklearn.log_model(
        model,
        name="model",
        signature=signature,
        input_example=sample.head(3),
        pyfunc_predict_fn="predict_proba",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Обучение вариантов модели")
    parser.add_argument("variants", nargs="+", choices=sorted(VARIANTS))
    args = parser.parse_args()

    setup_mlflow(EXPERIMENT)
    splits = split_by_domain(load_dataset())
    for name in args.variants:
        run_variant(name, VARIANTS[name], splits)


if __name__ == "__main__":
    main()
