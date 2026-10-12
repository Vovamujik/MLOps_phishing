"""Model Registry: регистрация модели из рана и перенос alias.

uv run python -m training.registry register <run_id> - новая версия из модели рана
uv run python -m training.registry alias champion <version> - alias указывает на версию
uv run python -m training.registry show - версии и их alias
"""

import argparse

import mlflow
from mlflow import MlflowClient
from mlflow.entities.model_registry import ModelVersion

from training.tracking import setup_mlflow
from training.train import EXPERIMENT

MODEL_NAME = "phishing-url-clf"
DESCRIPTION = "Классификатор фишинговых URL. Вход: колонка URL, выход: вероятности [легит, фишинг]."


def register(run_id: str) -> ModelVersion:
    """Создаёт новую версию из модели, залогированной в ране."""
    run = mlflow.get_run(run_id)
    # Без experiment_ids поиск идёт только по активному эксперименту
    models = mlflow.search_logged_models(
        experiment_ids=[run.info.experiment_id],
        filter_string=f"source_run_id = '{run_id}'",
        output_format="list",
    )
    if len(models) != 1:
        raise SystemExit(f"У рана {run_id} должна быть ровно одна модель, найдено {len(models)}")

    version = mlflow.register_model(f"models:/{models[0].model_id}", MODEL_NAME)

    client = MlflowClient()
    tags = {
        "variant": run.data.tags.get("variant", ""),
        "val_pr_auc": f"{run.data.metrics['val_pr_auc']:.4f}",
        "git_commit": run.data.tags.get("mlflow.source.git.commit", ""),
        "dataset_sha256": run.data.params.get("dataset_sha256", ""),
    }
    for key, value in tags.items():
        client.set_model_version_tag(MODEL_NAME, version.version, key, value)
    client.update_registered_model(MODEL_NAME, description=DESCRIPTION)
    return version


def set_alias(alias: str, version: str) -> None:
    """Ставит alias на версию. Если он уже стоял на другой версии, переезжает."""
    MlflowClient().set_registered_model_alias(MODEL_NAME, alias, version)


def show() -> None:
    client = MlflowClient()
    aliases_by_version: dict[str, list[str]] = {}
    for alias, number in client.get_registered_model(MODEL_NAME).aliases.items():
        aliases_by_version.setdefault(str(number), []).append(alias)

    for version in client.search_model_versions(f"name = '{MODEL_NAME}'"):
        aliases = ", ".join(aliases_by_version.get(version.version, [])) or "—"
        variant = version.tags.get("variant", "?")
        print(
            f"v{version.version}  alias: {aliases:<12} variant: {variant:<12} "
            f"val_pr_auc: {version.tags.get('val_pr_auc', '?')}  run: {version.run_id}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Работа с Model Registry")
    commands = parser.add_subparsers(dest="command", required=True)
    register_cmd = commands.add_parser("register", help="зарегистрировать модель рана")
    register_cmd.add_argument("run_id")
    alias_cmd = commands.add_parser("alias", help="поставить alias на версию")
    alias_cmd.add_argument("alias")
    alias_cmd.add_argument("version")
    commands.add_parser("show", help="показать версии")
    args = parser.parse_args()

    setup_mlflow(EXPERIMENT)
    if args.command == "register":
        version = register(args.run_id)
        print(f"Зарегистрирована версия {version.version} модели {MODEL_NAME}")
    elif args.command == "alias":
        set_alias(args.alias, args.version)
        print(f"{MODEL_NAME}@{args.alias} → v{args.version}")
    show()


if __name__ == "__main__":
    main()
