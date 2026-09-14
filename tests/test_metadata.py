"""Тесты метаданных сборки."""

from importlib.metadata import PackageNotFoundError

from mlops_phishing import metadata

from .conftest import read_pyproject_version


def raise_not_found(name):
    """Изобразить отсутствие установленного пакета."""
    raise PackageNotFoundError(name)


def test_version_matches_pyproject():
    """Версия в рантайме равна [project].version — источник правды один.

    Главный тест против ловушки с дублированием версии: если кто-то захардкодит
    версию в коде и забудет синхронизировать с pyproject.toml, тест покраснеет.
    """
    assert metadata.get_version() == read_pyproject_version()


def test_version_is_read_once(monkeypatch):
    """Повторные вызовы не сканируют файловую систему заново.

    Подменяем через monkeypatch, а не присваиванием: monkeypatch сам вернёт
    исходное значение после теста. Ручное присваивание с последующим `del`
    удалило бы имя из модуля насовсем и сломало следующие тесты.
    """
    calls = []

    def counting_version(name):
        calls.append(name)
        return "1.2.3"

    monkeypatch.setattr(metadata, "distribution_version", counting_version)
    metadata.get_version.cache_clear()

    assert metadata.get_version() == "1.2.3"
    assert metadata.get_version() == "1.2.3"
    assert len(calls) == 1


def test_version_falls_back_to_env_when_package_not_installed(monkeypatch):
    """Запуск из исходников без установки пакета берёт версию из APP_VERSION."""

    monkeypatch.setattr(metadata, "distribution_version", raise_not_found)
    monkeypatch.setenv("APP_VERSION", "9.9.9")
    metadata.get_version.cache_clear()

    assert metadata.get_version() == "9.9.9"


def test_version_is_unknown_without_package_and_env(monkeypatch):
    """Эндпоинт версии не должен падать, даже если версию взять неоткуда."""

    monkeypatch.setattr(metadata, "distribution_version", raise_not_found)
    monkeypatch.delenv("APP_VERSION", raising=False)
    metadata.get_version.cache_clear()

    assert metadata.get_version() == metadata.UNKNOWN


def test_build_info_defaults_to_unknown(monkeypatch):
    """Вне собранного образа коммит и время сборки неизвестны, но не падают."""
    monkeypatch.delenv("GIT_COMMIT", raising=False)
    monkeypatch.delenv("BUILD_TIME", raising=False)

    assert metadata.get_git_commit() == metadata.UNKNOWN
    assert metadata.get_build_time() == metadata.UNKNOWN


def test_build_info_comes_from_environment(monkeypatch):
    """В образе коммит и время сборки прокидываются через build-arg -> env."""
    monkeypatch.setenv("GIT_COMMIT", "a1b2c3d")
    monkeypatch.setenv("BUILD_TIME", "2026-09-14T17:00:00Z")

    assert metadata.get_git_commit() == "a1b2c3d"
    assert metadata.get_build_time() == "2026-09-14T17:00:00Z"
