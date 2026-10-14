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
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from mlops_phishing.features import normalize_urls, url_features
from training.data import SEED, TARGET, Splits, load_dataset, log_splits, split_by_domain
from training.evaluation import THRESHOLD, compute_metrics, log_diagnostics
from training.tracking import setup_mlflow

EXPERIMENT = "phishing-url-clf"


def tfidf_logreg(ngram_min: int, ngram_max: int, min_df: int, c: float) -> Pipeline:
    """TF-IDF по символьным n-граммам URL, затем логистическая регрессия.

    c — обратная сила регуляризации (C в sklearn).
    """
    return _url_logreg(_tfidf(ngram_min, ngram_max, min_df), c)


def normalized_tfidf_logreg(ngram_min: int, ngram_max: int, min_df: int, c: float) -> Pipeline:
    """Как tfidf_logreg, но перед TF-IDF из URL убираются схема, www. и / в конце.

    В датасете все легитимные URL начинаются с https://www. и не имеют пути,
    и без нормализации модель учит этот формат, а не признаки фишинга.
    """
    url = Pipeline(
        [
            ("normalize", FunctionTransformer(normalize_urls, feature_names_out="one-to-one")),
            ("tfidf", _tfidf(ngram_min, ngram_max, min_df)),
        ]
    )
    return _url_logreg(url, c)


def _tfidf(ngram_min: int, ngram_max: int, min_df: int) -> TfidfVectorizer:
    return TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(ngram_min, ngram_max),
        min_df=min_df,
        sublinear_tf=True,
        dtype=np.float32,
    )


def _url_logreg(url_features: TfidfVectorizer | Pipeline, c: float) -> Pipeline:
    return Pipeline(
        [
            ("features", ColumnTransformer([("url", url_features, "URL")])),
            ("classifier", LogisticRegression(C=c, max_iter=1000)),
        ]
    )


def handcrafted_hgb(learning_rate: float, max_iter: int, max_leaf_nodes: int) -> Pipeline:
    """Свои признаки URL (длина, доли символов, поддомены, IP, слова), затем бустинг."""
    return Pipeline(
        [
            ("features", FunctionTransformer(url_features)),
            (
                "classifier",
                HistGradientBoostingClassifier(
                    learning_rate=learning_rate,
                    max_iter=max_iter,
                    max_leaf_nodes=max_leaf_nodes,
                    random_state=SEED,
                ),
            ),
        ]
    )


@dataclass(frozen=True)
class Variant:
    build: Callable[..., Pipeline]
    params: dict[str, object]
    # Колонки, которые модель получает на вход. Для сервиса это только URL
    columns: tuple[str, ...] = ("URL",)
    # Свои функции в пайплайне skops без явного разрешения не сохраняет
    skops_trusted_types: tuple[str, ...] = ()


VARIANTS: dict[str, Variant] = {
    "baseline": Variant(tfidf_logreg, {"ngram_min": 3, "ngram_max": 5, "min_df": 10, "c": 1.0}),
    # Те же параметры, что у baseline: меняется только нормализация URL
    "normalized": Variant(
        normalized_tfidf_logreg,
        {"ngram_min": 3, "ngram_max": 5, "min_df": 10, "c": 1.0},
        skops_trusted_types=("mlops_phishing.features.normalize_urls",),
    ),
    "handcrafted": Variant(
        handcrafted_hgb,
        {"learning_rate": 0.1, "max_iter": 300, "max_leaf_nodes": 31},
        # Деревья HGB skops считает недоверенными; модель обучаем сами, источнику доверяем
        skops_trusted_types=(
            "mlops_phishing.features.url_features",
            "sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor",
        ),
    ),
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
        log_model(model, splits.val[columns], variant.skops_trusted_types)


def _predict(model: Pipeline, part: pd.DataFrame, columns: list[str]) -> np.ndarray:
    return model.predict_proba(part[columns])[:, 1]


def _prefixed(prefix: str, metrics: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def log_model(
    model: Pipeline, sample: pd.DataFrame, skops_trusted_types: tuple[str, ...] = ()
) -> None:
    sample = sample.head(100)
    signature = infer_signature(sample, model.predict_proba(sample))
    mlflow.sklearn.log_model(
        model,
        name="model",
        signature=signature,
        input_example=sample.head(3),
        pyfunc_predict_fn="predict_proba",
        skops_trusted_types=list(skops_trusted_types) or None,
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
