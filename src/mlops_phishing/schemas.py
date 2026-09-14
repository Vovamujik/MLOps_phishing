"""Pydantic-контракты ответов API.

Схемы вынесены отдельно от роутеров: они описывают публичный контракт, по ним
генерируется OpenAPI, и их же используют тесты. Роутер должен оставаться тонким.
"""

from enum import StrEnum

from pydantic import BaseModel, Field


class LivenessResponse(BaseModel):
    """Ответ liveness-проверки `/healthz`."""

    status: str = "ok"


class VersionResponse(BaseModel):
    """Ответ `/api/v1/version`: версия приложения и метаданные сборки."""

    name: str = Field(description="Имя приложения.")
    version: str = Field(description="Версия из [project].version в pyproject.toml.")
    git_commit: str = Field(description="SHA коммита, из которого собран образ.")
    build_time: str = Field(description="Время сборки образа, ISO-8601.")


class ComponentStatus(StrEnum):
    """Статус отдельного third-party компонента."""

    healthy = "healthy"
    unavailable = "unavailable"


class ReportStatus(StrEnum):
    """Сводный статус приложения."""

    ok = "ok"
    degraded = "degraded"


class ComponentHealth(BaseModel):
    """Здоровье одного компонента: версия, время ответа, причина отказа."""

    status: ComponentStatus
    version: str | None = Field(default=None, description="Версия компонента, если доступна.")
    latency_ms: float = Field(description="Время ответа компонента, миллисекунды.")
    error: str | None = Field(
        default=None, description="Причина отказа, если компонент недоступен."
    )


class HealthReport(BaseModel):
    """Ответ `/api/v1/health`: end-to-end состояние системы."""

    status: ReportStatus
    version: str = Field(description="Версия приложения.")
    environment: str = Field(description="Окружение: development / staging / production.")
    total_latency_ms: float = Field(description="Полное время сбора отчёта, миллисекунды.")
    components: dict[str, ComponentHealth] = Field(description="Состояние каждой зависимости.")
