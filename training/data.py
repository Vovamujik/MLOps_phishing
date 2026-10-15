"""Загрузка, очистка и разбиение датасета PhiUSIIL, регистрация частей в MLflow."""

from dataclasses import dataclass

import mlflow
import numpy as np
import pandas as pd
from mlflow.data.pandas_dataset import from_pandas
from sklearn.model_selection import GroupShuffleSplit

from training.download_data import CSV_PATH, CSV_SHA256, DATASET_URL, sha256_of

TARGET = "is_phishing"
SEED = 42
VAL_SIZE = 0.15
TEST_SIZE = 0.15


@dataclass(frozen=True)
class Splits:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame

    def named(self) -> list[tuple[str, pd.DataFrame]]:
        return [("train", self.train), ("val", self.val), ("test", self.test)]


def load_dataset() -> pd.DataFrame:
    """Читает проверенный CSV. Целевая колонка is_phishing: 1 - фишинг."""
    if not CSV_PATH.exists() or sha256_of(CSV_PATH) != CSV_SHA256:
        raise SystemExit("Нет CSV или он изменён: uv run python -m training.download_data")

    df = pd.read_csv(CSV_PATH)
    # FILENAME выдаёт источник (PhishTank, OpenPhish, ...), а значит и метку
    df = df.drop(columns=["FILENAME"]).drop_duplicates(subset="URL")
    # В датасете label=1 — легитимный сайт, а метрики нам нужны по фишингу
    df[TARGET] = 1 - df.pop("label")
    return df.reset_index(drop=True)


def split_by_domain(df: pd.DataFrame, seed: int = SEED) -> Splits:
    """Делит примерно 70/15/15 так, что каждый домен целиком попадает в одну часть.

    Иначе модель видит сайт в train и узнаёт его в test, и метрики завышены.
    """
    rest_idx, test_idx = _split_groups(df, TEST_SIZE, seed)
    rest = df.iloc[rest_idx]
    # Доля val от остатка, чтобы от всего датасета вышло VAL_SIZE
    train_idx, val_idx = _split_groups(rest, VAL_SIZE / (1 - TEST_SIZE), seed)
    return Splits(train=rest.iloc[train_idx], val=rest.iloc[val_idx], test=df.iloc[test_idx])


def _split_groups(df: pd.DataFrame, size: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    splitter = GroupShuffleSplit(n_splits=1, test_size=size, random_state=seed)
    return next(splitter.split(df, groups=df["Domain"]))


def log_splits(splits: Splits) -> None:
    """Регистрирует части как входные датасеты активного рана MLflow."""
    contexts = {"train": "training", "val": "validation", "test": "test"}
    for name, part in splits.named():
        dataset = from_pandas(part, source=DATASET_URL, name=f"phiusiil-{name}", targets=TARGET)
        mlflow.log_input(dataset, context=contexts[name])

    # Digest MLflow считается только по первым 10 000 строк, поэтому полный хеш файла пишем отдельно
    mlflow.log_params(
        {"dataset_sha256": CSV_SHA256, "split_seed": SEED}
        | {f"{name}_rows": len(part) for name, part in splits.named()}
    )


def main() -> None:
    splits = split_by_domain(load_dataset())
    for name, part in splits.named():
        print(
            f"{name:<5} {len(part):>7} строк, фишинг {part[TARGET].mean():.1%}, "
            f"доменов {part['Domain'].nunique()}"
        )
    shared = set(splits.train["Domain"]) & set(splits.test["Domain"])
    print(f"Общих доменов у train и test: {len(shared)}")


if __name__ == "__main__":
    main()
