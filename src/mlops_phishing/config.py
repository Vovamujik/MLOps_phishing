"""Настройки приложения из переменных окружения.

pydantic-settings собирает `Settings` из env и файла `.env`, попутно валидируя
типы: `DEBUG=yes` станет `True`, а `HEALTH_CHECK_TIMEOUT=abc` уронит запуск
с внятной ошибкой, а не всплывёт багом в рантайме.
"""

from functools import cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Конфигурация приложения.

    Любое поле переопределяется переменной окружения того же имени в верхнем
    регистре: `LOG_LEVEL=DEBUG uv run uvicorn mlops_phishing.app:app`.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "mlops-phishing"

    environment: Literal["development", "staging", "production"] = "development"

    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    debug: bool = False

    database_url: str = Field(
        default="postgresql://postgres:postgres@localhost:5432/mlops_phishing",
        description="Строка подключения к Postgres в формате asyncpg.",
    )

    health_check_timeout: float = Field(
        default=3.0,
        gt=0,
        description="Таймаут одной проверки зависимости в /api/v1/health, секунды.",
    )


@cache
def get_settings() -> Settings:
    """Вернуть настройки приложения (читаются из окружения один раз).

    Кэш нужен, чтобы не перечитывать .env на каждый запрос и чтобы все части
    приложения видели одну и ту же конфигурацию. В тестах, где окружение
    подменяется, кэш сбрасывается через `get_settings.cache_clear()`.
    """
    return Settings()
