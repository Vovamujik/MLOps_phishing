"""Метаданные сборки: версия приложения, git-коммит, время сборки.

Единственный источник правды о версии — `[project].version` в pyproject.toml.
Оттуда её берёт сборочный бэкенд и кладёт в `.dist-info/METADATA` собранного
пакета, а `importlib.metadata` читает уже оттуда.
"""

import os
from functools import cache
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as distribution_version

DISTRIBUTION_NAME = "mlops-phishing"

UNKNOWN = "unknown"


@cache
def get_version() -> str:
    """Вернуть версию приложения.

    `importlib.metadata` сканирует файловую систему, а эндпоинт версии может
    дёргаться часто — поэтому результат кэшируется: в пределах процесса версия
    измениться не может.

    Фолбэк на `APP_VERSION` нужен для запуска из исходников без установки
    пакета (редкий случай, но падать на этом эндпоинт не должен).
    """
    try:
        return distribution_version(DISTRIBUTION_NAME)
    except PackageNotFoundError:
        return os.getenv("APP_VERSION", UNKNOWN)


def get_git_commit() -> str:
    """Вернуть sha коммита, из которого собран образ."""
    return os.getenv("GIT_COMMIT", UNKNOWN)


def get_build_time() -> str:
    """Вернуть время сборки образа в формате ISO-8601."""
    return os.getenv("BUILD_TIME", UNKNOWN)
