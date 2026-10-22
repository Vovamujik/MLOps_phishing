"""Pydantic-контракты ответов API.

Схемы вынесены отдельно от роутеров: они описывают публичный контракт, по ним
генерируется OpenAPI, и их же используют тесты. Роутер должен оставаться тонким.
"""

import uuid
from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from mlops_phishing.tasks_repo import TaskStatus


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


class TaskCreate(BaseModel):
    """Тело `POST /api/v1/tasks`: URL, который нужно проверить."""

    model_config = ConfigDict(str_strip_whitespace=True)

    url: str = Field(
        min_length=1,
        max_length=2048,
        # Без пробелов и управляющих символов: NUL база не примет уже после публикации
        pattern=r"(?i)^https?://[^\s\x00-\x1f\x7f]+$",
        description="Полный URL со схемой http или https. Сервис его не открывает.",
        examples=["http://paypal-secure-login.verify-account.xyz/signin"],
    )


class TaskAccepted(BaseModel):
    """Ответ 202 на `POST /api/v1/tasks`: задача в очереди, результата ещё нет."""

    task_id: uuid.UUID
    status: TaskStatus = Field(description="PENDING: задача ждёт воркер.")
    status_url: str = Field(
        description="Где смотреть результат; тот же адрес в заголовке Location."
    )


class TaskView(BaseModel):
    """Ответ `GET /api/v1/tasks/{task_id}`: статус задачи и результат, когда он готов."""

    model_config = ConfigDict(from_attributes=True)

    task_id: uuid.UUID
    url: str
    source: str = Field(description="Откуда задача: api или batch (источник из CSV).")
    status: TaskStatus
    is_phishing: bool | None = Field(
        description="Вероятность не ниже порога; null, пока не готово."
    )
    phishing_probability: float | None = Field(description="Вероятность фишинга от 0 до 1.")
    model_version: str | None = Field(description="Версия модели, которая посчитала результат.")
    enqueued_at: datetime = Field(description="Когда задача принята брокером.")
    completed_at: datetime | None = Field(description="Когда воркер записал результат.")
