"""Метрики и диагностические артефакты. Позитивный класс — фишинг."""

import mlflow
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from training.data import TARGET

THRESHOLD = 0.5
TOP_N = 20


def compute_metrics(y_true: pd.Series, proba: np.ndarray) -> dict[str, float]:
    pred = proba >= THRESHOLD
    tn, fp, _, _ = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        # Основная: не зависит от порога и показывает, как модель ранжирует фишинг
        "pr_auc": average_precision_score(y_true, proba),
        "roc_auc": roc_auc_score(y_true, proba),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred),
        "f1": f1_score(y_true, pred),
        # Доля легитимных сайтов, которые мы зря назвали фишингом
        "fpr": fp / (fp + tn),
        "accuracy": accuracy_score(y_true, pred),
    }


def log_diagnostics(model: Pipeline, test: pd.DataFrame, proba: np.ndarray) -> None:
    """Графики по test и таблицы ошибок, чтобы видеть, где и почему модель ошибается."""
    y_true = test[TARGET]

    fig, ax = _figure()
    ConfusionMatrixDisplay.from_predictions(
        y_true, proba >= THRESHOLD, display_labels=["legit", "phishing"], ax=ax, colorbar=False
    )
    mlflow.log_figure(fig, "plots/confusion_matrix.png")

    fig, ax = _figure()
    RocCurveDisplay.from_predictions(y_true, proba, ax=ax)
    mlflow.log_figure(fig, "plots/roc_curve.png")

    fig, ax = _figure()
    PrecisionRecallDisplay.from_predictions(y_true, proba, ax=ax)
    mlflow.log_figure(fig, "plots/pr_curve.png")

    scored = test[["URL", TARGET]].assign(phishing_proba=proba)
    missed = scored[scored[TARGET] == 1].nsmallest(TOP_N, "phishing_proba")
    false_alarms = scored[scored[TARGET] == 0].nlargest(TOP_N, "phishing_proba")
    mlflow.log_table(missed, "errors/missed_phishing.json")
    mlflow.log_table(false_alarms, "errors/false_alarms.json")

    _log_top_features(model)


def _log_top_features(model: Pipeline) -> None:
    """Для линейных моделей: признаки с самым большим весом в сторону каждого класса."""
    classifier = model[-1]
    if not hasattr(classifier, "coef_"):
        return
    weights = pd.Series(classifier.coef_[0], index=model[:-1].get_feature_names_out())
    top = pd.concat([weights.nlargest(TOP_N), weights.nsmallest(TOP_N)])
    table = top.rename("weight").rename_axis("feature").reset_index()
    mlflow.log_table(table, "top_features.json")


def _figure() -> tuple[Figure, Axes]:
    fig = Figure(figsize=(5, 4), layout="constrained")
    return fig, fig.subplots()
