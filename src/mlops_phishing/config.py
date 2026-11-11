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

    mlflow_tracking_uri: str = Field(
        default="http://localhost:5050",
        description="Адрес MLflow, из реестра которого модель загружается при старте.",
    )
    model_name: str = Field(default="phishing-url-clf", description="Имя модели в реестре.")
    model_alias: str = Field(
        default="champion",
        description="Alias версии, которую загружает воркер. Смена — без правки кода.",
    )
    phishing_threshold: float = Field(
        default=0.5,
        ge=0,
        le=1,
        description="Порог вероятности, начиная с которого URL считается фишингом.",
    )

    rabbitmq_url: str = Field(
        default="amqp://app:app@localhost:5672/",
        description="Подключение к RabbitMQ (AMQP). Логин и пароль из него же идут в HTTP API.",
    )
    rabbitmq_management_url: str = Field(
        default="http://localhost:15672",
        description="HTTP API управления RabbitMQ: через него задаётся топология очередей.",
    )
    # 0 — очередь без лимита и без nack: режим «без противодавления», только для эксперимента
    tasks_queue_max_length: int = Field(
        default=100,
        ge=0,
        description="Лимит готовых сообщений в очереди задач; сверх него брокер отвечает nack.",
    )
    tasks_delivery_limit: int = Field(
        default=3,
        gt=0,
        description="Сколько повторных доставок даётся задаче, прежде чем она уйдёт в DLQ.",
    )
    # Отказ при полной очереди приходит за миллисекунды. Таймаут нужен, если брокер завис
    # или недоступен: клиент получает 503, а не висит
    publish_timeout: float = Field(
        default=2.0,
        gt=0,
        description="Сколько API ждёт брокер (соединение и подтверждение публикации), секунды.",
    )

    worker_pool_size: int = Field(
        default=4,
        gt=0,
        description="Потоков для inference в одном воркере: столько задач считается одновременно.",
    )
    # prefetch = размер пула: каждой взятой задаче сразу есть свободный поток.
    # Больше — лишние задачи ждут в памяти воркера, и второй воркер их не заберёт;
    # меньше — потоки простаивают. 0 — без ограничения, только для эксперимента
    worker_prefetch: int = Field(
        default=4,
        ge=0,
        description="Сколько неподтверждённых сообщений брокер отдаёт одному воркеру.",
    )
    inference_delay_seconds: float = Field(
        default=0.2,
        ge=0,
        description="Добавочное время inference: имитация тяжёлой модели для экспериментов.",
    )
    worker_graceful_timeout: float = Field(
        default=30.0,
        gt=0,
        description="Сколько при остановке ждать задачи, уже взятые в работу, секунды.",
    )

    # Первая пауза короткая, но не нулевая (не busy loop). После отказа очередь снова принимает
    # только с 80% лимита; сколько она закрыта, удвоение находит само, без настройки под воркеры
    source_backoff_initial: float = Field(
        default=0.05,
        gt=0,
        description="Пауза источника после первого отказа брокера, секунды; дальше удваивается.",
    )
    # Пока источник спит, воркеры разбирают «пропускная способность × пауза» задач, а очередь
    # открывается с 80% лимита. Условие: T × потолок < 0.8 × лимит — при лимите 100 и 1 с
    # строго держится до 3 воркеров по 20 задач/с; на четырёх (80) — граница, запас даёт джиттер
    source_backoff_max: float = Field(
        default=1.0,
        gt=0,
        description="Потолок паузы источника между попытками, секунды.",
    )


@cache
def get_settings() -> Settings:
    """Вернуть настройки приложения (читаются из окружения один раз).

    Кэш нужен, чтобы не перечитывать .env на каждый запрос и чтобы все части
    приложения видели одну и ту же конфигурацию. В тестах, где окружение
    подменяется, кэш сбрасывается через `get_settings.cache_clear()`.
    """
    return Settings()
