"""Идемпотентность записи на настоящем Postgres (Testcontainers).

Гарантию даёт уникальный ключ в базе, а его поведение под конкурентной
записью заглушкой не проверить.
"""

import asyncio
import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime

import asyncpg
import pytest

from mlops_phishing import tasks_repo
from mlops_phishing.tasks_repo import TaskResult, TaskStatus

pytestmark = pytest.mark.integration

URL = "http://paypal-verify.example/login"
NOW = datetime(2026, 10, 20, 12, 0, tzinfo=UTC)
PHISHING = TaskResult(is_phishing=True, phishing_probability=0.97, model_version="2")
LEGIT = TaskResult(is_phishing=False, phishing_probability=0.03, model_version="1")


@pytest.fixture
async def pool(postgres_dsn) -> AsyncIterator[asyncpg.Pool]:
    pool = await asyncpg.create_pool(postgres_dsn, min_size=1, max_size=10)
    await tasks_repo.ensure_schema(pool)
    yield pool
    await pool.close()


async def test_schema_creation_is_repeatable(pool):
    await tasks_repo.ensure_schema(pool)


async def test_first_result_is_written(pool):
    task_id = uuid.uuid4()

    assert await tasks_repo.save_result(pool, task_id, URL, "batch", NOW, PHISHING) is True

    record = await tasks_repo.get_task(pool, task_id)
    assert record is not None
    assert record.status is TaskStatus.DONE
    assert (record.is_phishing, record.phishing_probability, record.model_version) == (
        True,
        0.97,
        "2",
    )
    assert record.completed_at is not None


async def test_redelivery_does_not_duplicate_or_overwrite_result(pool):
    """Повторная доставка той же задачи: строка одна, первый результат на месте."""
    task_id = uuid.uuid4()
    await tasks_repo.save_result(pool, task_id, URL, "batch", NOW, PHISHING)

    assert await tasks_repo.save_result(pool, task_id, URL, "batch", NOW, LEGIT) is False

    record = await tasks_repo.get_task(pool, task_id)
    assert record is not None
    assert record.model_version == "2"
    count = await pool.fetchval("SELECT count(*) FROM tasks WHERE task_id = $1", task_id)
    assert count == 1


async def test_concurrent_redeliveries_write_exactly_once(pool):
    """Десять воркеров одновременно с одной задачей: записал ровно один."""
    task_id = uuid.uuid4()

    written = await asyncio.gather(
        *(tasks_repo.save_result(pool, task_id, URL, "batch", NOW, PHISHING) for _ in range(10))
    )

    assert written.count(True) == 1
    count = await pool.fetchval("SELECT count(*) FROM tasks WHERE task_id = $1", task_id)
    assert count == 1


async def test_api_pending_task_becomes_done_and_keeps_its_source(pool):
    task_id = uuid.uuid4()
    await tasks_repo.insert_pending(pool, task_id, URL, NOW)

    assert await tasks_repo.save_result(pool, task_id, URL, "api", NOW, PHISHING) is True

    record = await tasks_repo.get_task(pool, task_id)
    assert record is not None
    assert (record.status, record.source) == (TaskStatus.DONE, "api")


async def test_late_pending_insert_does_not_roll_done_back(pool):
    """Воркер успел раньше API: PENDING не должен затереть готовый результат."""
    task_id = uuid.uuid4()
    await tasks_repo.save_result(pool, task_id, URL, "api", NOW, PHISHING)

    await tasks_repo.insert_pending(pool, task_id, URL, NOW)

    record = await tasks_repo.get_task(pool, task_id)
    assert record is not None
    assert record.status is TaskStatus.DONE


async def test_unknown_task_is_none(pool):
    assert await tasks_repo.get_task(pool, uuid.uuid4()) is None
