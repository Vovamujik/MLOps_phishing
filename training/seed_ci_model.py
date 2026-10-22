"""Игрушечная модель для CI: регистрирует champion в пустом MLflow.

В CI нет датасета и времени на настоящее обучение, а воркеру нужна версия
за alias, чтобы smoke-тест прошёл весь путь: реестр → загрузка при старте воркера →
задача через POST /api/v1/tasks → результат в базе.
Пайплайн тот же, что у baseline, поэтому формат и сигнатура как у настоящей модели.

Запуск: uv run python -m training.seed_ci_model
"""

import mlflow
import pandas as pd

from training import registry
from training.tracking import setup_mlflow
from training.train import log_model, tfidf_logreg

EXPERIMENT = "ci-seed"
ALIAS = "champion"

# 1 — фишинг, как is_phishing в настоящих данных
URLS = {
    "https://www.google.com": 0,
    "https://www.wikipedia.org": 0,
    "https://www.python.org": 0,
    "https://www.uni-mainz.de": 0,
    "http://paypal-secure-login.verify-account.xyz/signin": 1,
    "http://appleid.apple.com.account-verify.top/login": 1,
    "http://secure-bank-update.info/confirm?id=123": 1,
    "http://free-gift-card.win/claim": 1,
}


def main() -> None:
    setup_mlflow(EXPERIMENT)
    frame = pd.DataFrame({"URL": list(URLS)})
    model = tfidf_logreg(ngram_min=3, ngram_max=5, min_df=1, c=1.0)
    model.fit(frame, list(URLS.values()))

    with mlflow.start_run(run_name="ci-seed") as run:
        mlflow.set_tag("variant", "ci-seed")
        log_model(model, frame)

    version = registry.register(run.info.run_id)
    registry.set_alias(ALIAS, version.version)
    print(f"{registry.MODEL_NAME}@{ALIAS} → v{version.version} (игрушечная модель для CI)")


if __name__ == "__main__":
    main()
