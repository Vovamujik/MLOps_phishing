"""Задачи конвейера в Postgres: статус и результат, идемпотентная запись.

Повторная доставка сообщения не должна дублировать результат. Гарантию даёт
уникальный ключ task_id в самой базе: запись — один атомарный INSERT ... ON
CONFLICT. Схема «проверить, затем вставить» здесь не годится: между проверкой и
вставкой успел бы вклиниться второй воркер с той же задачей.
"""

import uuid
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

import asyncpg

SCHEMA = """
CREATE TABLE IF NOT EXISTS tasks (
    task_id uuid PRIMARY KEY,
    url text NOT NULL,
    source text NOT NULL,
    status text NOT NULL CHECK (status IN ('PENDING', 'DONE')),
    is_phishing boolean,
    phishing_probability double precision,
    model_version text,
    enqueued_at timestamptz NOT NULL,
    completed_at timestamptz
)
"""


class TaskStatus(StrEnum):
    PENDING = "PENDING"
    DONE = "DONE"


@dataclass(frozen=True)
class TaskResult:
    """Что посчитал воркер."""

    is_phishing: bool
    phishing_probability: float
    model_version: str


@dataclass(frozen=True)
class TaskRecord:
    task_id: uuid.UUID
    url: str
    source: str
    status: TaskStatus
    is_phishing: bool | None
    phishing_probability: float | None
    model_version: str | None
    enqueued_at: datetime
    completed_at: datetime | None


async def ensure_schema(pool: asyncpg.Pool) -> None:
    """Создать таблицу, если её нет. Безопасно при каждом старте."""
    await pool.execute(SCHEMA)


async def insert_pending(
    pool: asyncpg.Pool, task_id: uuid.UUID, url: str, enqueued_at: datetime
) -> None:
    """Задача от API, уже принятая брокером.

    DO NOTHING при конфликте: воркер мог успеть раньше, и его DONE нельзя
    откатить назад в PENDING.
    """
    await pool.execute(
        """
        INSERT INTO tasks (task_id, url, source, status, enqueued_at)
        VALUES ($1, $2, 'api', 'PENDING', $3)
        ON CONFLICT (task_id) DO NOTHING
        """,
        task_id,
        url,
        enqueued_at,
    )


async def save_result(
    pool: asyncpg.Pool,
    task_id: uuid.UUID,
    url: str,
    source: str,
    enqueued_at: datetime,
    result: TaskResult,
) -> bool:
    """Записать результат. False — задача уже была готова (повторная доставка).

    Первый результат остаётся: условие WHERE не даёт повтору его перезаписать.
    """
    status = await pool.execute(
        """
        INSERT INTO tasks (task_id, url, source, status, is_phishing,
                           phishing_probability, model_version, enqueued_at, completed_at)
        VALUES ($1, $2, $3, 'DONE', $4, $5, $6, $7, now())
        ON CONFLICT (task_id) DO UPDATE SET
            status = 'DONE',
            is_phishing = EXCLUDED.is_phishing,
            phishing_probability = EXCLUDED.phishing_probability,
            model_version = EXCLUDED.model_version,
            completed_at = EXCLUDED.completed_at
        WHERE tasks.status <> 'DONE'
        """,
        task_id,
        url,
        source,
        result.is_phishing,
        result.phishing_probability,
        result.model_version,
        enqueued_at,
    )
    # asyncpg возвращает тег команды: "INSERT 0 1" — записали, "INSERT 0 0" — уже было
    return status.endswith(" 1")


async def get_task(pool: asyncpg.Pool, task_id: uuid.UUID) -> TaskRecord | None:
    row = await pool.fetchrow("SELECT * FROM tasks WHERE task_id = $1", task_id)
    if row is None:
        return None
    return TaskRecord(**{**dict(row), "status": TaskStatus(row["status"])})
