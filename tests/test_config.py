"""Тесты конфигурации приложения."""

import pytest
from pydantic import ValidationError

from mlops_phishing.config import Settings, get_settings


def test_defaults_allow_startup_without_environment():
    """Без единой переменной окружения приложение должно подниматься."""
    settings = Settings()

    assert settings.app_name == "mlops-phishing"
    assert settings.environment == "development"
    assert settings.log_level == "INFO"
    assert settings.debug is False
    assert settings.health_check_timeout == 3.0


def test_environment_variables_override_defaults(monkeypatch):
    """Значения из окружения побеждают дефолты и приводятся к нужным типам."""
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("DEBUG", "true")
    monkeypatch.setenv("HEALTH_CHECK_TIMEOUT", "1.5")
    monkeypatch.setenv("DATABASE_URL", "postgresql://user:pass@db:5432/app")

    settings = Settings()

    assert settings.log_level == "DEBUG"
    assert settings.debug is True
    assert settings.health_check_timeout == 1.5
    assert settings.database_url == "postgresql://user:pass@db:5432/app"


def test_settings_have_no_version_field():
    """Версия не дублируется в конфиге: её источник — metadata.get_version().

    Тест фиксирует архитектурное решение. Если кто-то добавит сюда поле
    version, появится второй источник правды — и он об этом узнает сразу.
    """
    assert "version" not in Settings.model_fields


def test_invalid_environment_is_rejected(monkeypatch):
    """Опечатка в ENVIRONMENT роняет запуск, а не всплывает в health-отчёте."""
    monkeypatch.setenv("ENVIRONMENT", "prodaction")

    with pytest.raises(ValidationError, match="environment"):
        Settings()


def test_invalid_log_level_is_rejected(monkeypatch):
    """LOG_LEVEL принимает только реальные уровни логирования."""
    monkeypatch.setenv("LOG_LEVEL", "VERBOSE")

    with pytest.raises(ValidationError, match="log_level"):
        Settings()


def test_non_positive_health_timeout_is_rejected(monkeypatch):
    """Нулевой таймаут проверки зависимости бессмыслен и запрещён."""
    monkeypatch.setenv("HEALTH_CHECK_TIMEOUT", "0")

    with pytest.raises(ValidationError, match="health_check_timeout"):
        Settings()


def test_unknown_environment_variables_are_ignored(monkeypatch):
    """Чужие переменные окружения контейнера не должны ронять приложение."""
    monkeypatch.setenv("KUBERNETES_SERVICE_HOST", "10.0.0.1")
    monkeypatch.setenv("SOME_UNRELATED_VAR", "whatever")

    Settings()


def test_get_settings_returns_same_instance():
    """Все части приложения видят одну и ту же конфигурацию."""
    assert get_settings() is get_settings()
