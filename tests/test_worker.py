"""Тесты воркера на TestRabbitBroker: подтверждения, повторы, DLQ, event loop.

TestRabbitBroker — брокер в памяти: очереди, prefetch и лимита в нём нет. Поэтому
здесь проверяется логика обработчика: когда ack, когда reject и с возвратом ли в
очередь. Поведение настоящего брокера — в интеграционных тестах.
"""

import asyncio
import threading
import time
import uuid
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from unittest import mock

import pytest
from faststream import TestApp
from faststream.rabbit import TestRabbitBroker
from faststream.rabbit.message import RabbitMessage
from pydantic import ValidationError

from mlops_phishing import ml, tasks_repo
from mlops_phishing import worker as worker_module
from mlops_phishing.config import Settings
from mlops_phishing.messaging import TASKS_QUEUE
from mlops_phishing.worker import Worker, build_worker, score

from .conftest import FakeClassifier, make_model


def task_message(**overrides: object) -> dict[str, object]:
    message: dict[str, object] = {
        "task_id": str(uuid.uuid4()),
        "url": "http://paypal-verify.example/login",
        "source": "batch",
        "enqueued_at": datetime.now(UTC).isoformat(),
    }
    return message | overrides


@pytest.fixture
def events() -> list[str]:
    """Порядок событий: запись результата и подтверждения брокеру."""
    return []


@pytest.fixture
def acks(events: list[str]) -> Iterator[None]:
    # Патчим класс из faststream.rabbit.message: публичное имя — Annotated-алиас
    with (
        mock.patch.object(RabbitMessage, "ack", autospec=True) as ack,
        mock.patch.object(RabbitMessage, "reject", autospec=True) as reject,
    ):
        ack.side_effect = lambda *args, **kwargs: events.append("ack")
        reject.side_effect = lambda *args, **kwargs: events.append(f"reject{kwargs}")
        yield


@pytest.fixture
def worker() -> Iterator[Worker]:
    """Воркер с уже «загруженными» моделью, пулом потоков и базой."""
    built = build_worker(Settings(inference_delay_seconds=0, worker_pool_size=2))
    executor = ThreadPoolExecutor(2)
    built.state.model = make_model(FakeClassifier(phishing_probability=0.9), version="2")
    built.state.executor = executor
    built.state.db_pool = mock.Mock()
    yield built
    executor.shutdown(wait=True)


def save_result_stub(events: list[str], written: bool = True) -> mock.AsyncMock:
    async def save(*args: object) -> bool:
        events.append("save")
        return written

    return mock.AsyncMock(side_effect=save)


@pytest.mark.usefixtures("acks")
async def test_ack_comes_after_result_is_saved(worker, events, monkeypatch):
    """Ack до записи потерял бы задачу, упади воркер между ними."""
    save = save_result_stub(events)
    monkeypatch.setattr(tasks_repo, "save_result", save)
    message = task_message()

    async with TestRabbitBroker(worker.broker) as broker:
        await broker.publish(message, queue=TASKS_QUEUE)

    assert events == ["save", "ack"]
    _, task_id, url, source, _, result = save.call_args.args
    assert (str(task_id), url, source) == (message["task_id"], message["url"], "batch")
    assert (result.is_phishing, result.phishing_probability, result.model_version) == (
        True,
        0.9,
        "2",
    )
    assert worker.state.stats.processed == 1


@pytest.mark.usefixtures("acks")
async def test_database_failure_rejects_with_requeue(worker, events, monkeypatch):
    """reject(requeue=True) считается попыткой: после delivery-limit задача уйдёт в DLQ."""
    monkeypatch.setattr(tasks_repo, "save_result", mock.AsyncMock(side_effect=OSError("down")))

    async with TestRabbitBroker(worker.broker) as broker:
        await broker.publish(task_message(), queue=TASKS_QUEUE)

    assert events == ["reject{'requeue': True}"]
    assert worker.state.stats.retried == 1
    assert worker.state.stats.in_flight == 0


@pytest.mark.usefixtures("acks")
async def test_malformed_message_goes_to_dlq_without_retries(worker, events):
    """Битое сообщение повтор не исправит: reject без возврата, сразу в DLQ."""
    async with TestRabbitBroker(worker.broker) as broker:
        with pytest.raises(ValidationError):
            await broker.publish({"url": "no task id"}, queue=TASKS_QUEUE)

    assert events == ["reject{}"]


@pytest.mark.usefixtures("acks")
async def test_redelivery_of_done_task_is_acked_without_rewrite(worker, events, monkeypatch):
    monkeypatch.setattr(tasks_repo, "save_result", save_result_stub(events, written=False))

    async with TestRabbitBroker(worker.broker) as broker:
        await broker.publish(task_message(), queue=TASKS_QUEUE)

    assert events == ["save", "ack"]
    assert worker.state.stats.duplicates == 1
    assert worker.state.stats.processed == 0


@pytest.mark.usefixtures("acks")
async def test_inference_runs_in_pool_thread(worker, events, monkeypatch):
    classifier = FakeClassifier()
    worker.state.model = make_model(classifier)
    monkeypatch.setattr(tasks_repo, "save_result", save_result_stub(events))

    async with TestRabbitBroker(worker.broker) as broker:
        await broker.publish(task_message(), queue=TASKS_QUEUE)

    assert classifier.thread_ids
    assert classifier.thread_ids[0] != threading.get_ident()


@pytest.mark.usefixtures("acks")
async def test_event_loop_stays_free_during_inference(worker, events, monkeypatch):
    """Пока модель «думает» 0.3 с, event loop не замирает: значит, /health ответит."""
    worker.state.settings = Settings(inference_delay_seconds=0.3)
    monkeypatch.setattr(tasks_repo, "save_result", save_result_stub(events))
    gaps: list[float] = []

    async def ticker() -> None:
        last = time.perf_counter()
        while True:
            await asyncio.sleep(0.01)
            now = time.perf_counter()
            gaps.append(now - last)
            last = now

    tick = asyncio.create_task(ticker())
    async with TestRabbitBroker(worker.broker) as broker:
        await broker.publish(task_message(), queue=TASKS_QUEUE)
    tick.cancel()

    # Заблокированный loop дал бы один разрыв ~0.3 с вместо десятков коротких
    assert len(gaps) >= 10
    assert max(gaps) < 0.1


@pytest.mark.usefixtures("acks")
async def test_model_is_loaded_once_at_startup(events, monkeypatch):
    """Модель грузится в lifespan до первого сообщения, а не на каждое."""
    loads: list[str] = []

    def load(settings: Settings) -> ml.LoadedModel:
        loads.append(settings.model_alias)
        return make_model()

    pool = mock.AsyncMock()
    monkeypatch.setattr(ml, "load_model", load)
    monkeypatch.setattr(worker_module.asyncpg, "create_pool", mock.AsyncMock(return_value=pool))
    monkeypatch.setattr(tasks_repo, "save_result", save_result_stub(events))
    built = build_worker(Settings(inference_delay_seconds=0))

    async with TestRabbitBroker(built.broker) as broker, TestApp(built.app):
        assert loads == ["champion"]
        for _ in range(3):
            await broker.publish(task_message(), queue=TASKS_QUEUE)

    assert loads == ["champion"]
    assert events.count("ack") == 3
    pool.close.assert_awaited_once()


def test_prefetch_matches_pool_size_by_default():
    settings = Settings()

    assert settings.worker_prefetch == settings.worker_pool_size


def test_score_adds_configured_delay():
    model = make_model(FakeClassifier(phishing_probability=0.4))

    started = time.perf_counter()
    probability = score(model, "http://x.example", 0.05)

    assert probability == pytest.approx(0.4)
    assert time.perf_counter() - started >= 0.05
