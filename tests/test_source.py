"""Тесты источника: ленивое чтение, повтор после nack, паузы, остановка.

Отказ брокера при полной очереди изображает заглушка: TestRabbitBroker лимита
очереди не знает. Поведение настоящего RabbitMQ — в интеграционных тестах.
"""

import asyncio
import logging
import signal
import tracemalloc
from collections.abc import Iterable, Iterator
from pathlib import Path
from unittest import mock

import pytest
from aio_pika.exceptions import DeliveryError, PublishError
from aiormq.abc import DeliveredMessage
from faststream.rabbit import RabbitBroker, RabbitMessage, TestRabbitBroker
from pamqp.commands import Basic
from pamqp.header import ContentHeader

from mlops_phishing import source
from mlops_phishing.config import Settings
from mlops_phishing.messaging import PUBLISH_CHANNEL, TASKS_QUEUE, TaskMessage
from mlops_phishing.source import Backoff, Publish, Row, SourceStats, run_batch, task_id_for

BATCH = "batch-1"
HEADER = "FILENAME,URL,label\n"


def nack() -> DeliveryError:
    """Ответ брокера на публикацию в полную очередь (overflow=reject-publish)."""
    return DeliveryError(None, Basic.Nack(delivery_tag=1))


def unroutable() -> PublishError:
    """Ответ брокера, когда очереди нет: сообщение вернулось отправителю."""
    returned = Basic.Return(reply_code=312, reply_text="NO_ROUTE", exchange="", routing_key="tasks")
    message = DeliveredMessage(delivery=returned, header=ContentHeader(), body=b"", channel=None)
    return PublishError(message, returned)


class FakeBroker:
    """Брокер-заглушка: первые `full` попыток отвечает nack, будто очередь полна."""

    def __init__(self, full: int = 0, error: Exception | None = None) -> None:
        self.full = full
        self.error = error
        self.attempts: list[TaskMessage] = []
        self.accepted: list[TaskMessage] = []

    async def publish(self, task: TaskMessage) -> None:
        self.attempts.append(task)
        if self.error is not None:
            raise self.error
        if self.full > 0:
            self.full -= 1
            raise nack()
        self.accepted.append(task)


def make_rows(count: int) -> list[Row]:
    return [Row(number, f"http://login-verify{number}.example/") for number in range(1, count + 1)]


async def publish_rows(
    rows: Iterable[Row], publish: Publish, backoff: Backoff | None = None
) -> SourceStats:
    stats = SourceStats()
    backoff = backoff or Backoff(initial=0.001, maximum=0.004)
    # Таймаут: сломанный цикл повторов уронит тест, а не повесит прогон
    batch = run_batch(rows, publish, BATCH, backoff, asyncio.Event(), stats)
    await asyncio.wait_for(batch, timeout=5)
    return stats


def write_csv(path: Path, urls: Iterable[str]) -> Path:
    # BOM в начале, как у PhiUSIIL
    path.write_text("﻿" + HEADER + "".join(f"f.txt,{url},1\n" for url in urls))
    return path


def test_read_urls_streams_rows_after_bom(tmp_path):
    path = write_csv(tmp_path / "urls.csv", ["http://a.example/", "http://b.example/"])

    rows = source.read_urls(path)

    assert isinstance(rows, Iterator)
    assert list(rows) == [Row(1, "http://a.example/"), Row(2, "http://b.example/")]


def test_read_urls_does_not_load_whole_file(tmp_path):
    """50 000 строк, около 3 МБ: память при чтении не растёт с размером файла."""
    path = write_csv(tmp_path / "big.csv", (f"http://site{i}.example/path" for i in range(50_000)))

    tracemalloc.start()
    try:
        count = sum(1 for _ in source.read_urls(path))
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert count == 50_000
    assert path.stat().st_size > 1_500_000
    assert peak < 256 * 1024


def test_read_urls_requires_url_column(tmp_path):
    path = tmp_path / "no_url.csv"
    path.write_text("FILENAME,label\nf.txt,1\n")

    with pytest.raises(ValueError, match="URL"):
        next(source.read_urls(path))


def test_task_id_depends_only_on_batch_and_row():
    assert task_id_for(BATCH, 7) == task_id_for(BATCH, 7)
    assert task_id_for(BATCH, 7) != task_id_for(BATCH, 8)
    assert task_id_for(BATCH, 7) != task_id_for("batch-2", 7)


def test_backoff_doubles_with_jitter_up_to_cap():
    backoff = Backoff(initial=0.05, maximum=1.0)
    expected = [0.05, 0.1, 0.2, 0.4, 0.8, 1.0, 1.0]

    delays = [backoff.next_delay() for _ in expected]

    # Половина паузы гарантирована, вторая случайна: delay ∈ [d/2, d]
    for delay, cap in zip(delays, expected, strict=True):
        assert cap / 2 <= delay <= cap
    assert Backoff(initial=5, maximum=1).next_delay() <= 1


def test_backoff_never_overflows_and_resets():
    """Очередь полна час — пауза так и остаётся на потолке, без OverflowError."""
    backoff = Backoff(initial=0.05, maximum=1.0)

    for _ in range(5_000):
        assert backoff.next_delay() <= 1.0
    backoff.reset()

    # После сброса — снова начальная пауза, и не ноль
    assert 0.025 <= backoff.next_delay() <= 0.05


async def test_full_queue_retries_same_task_until_confirmed():
    broker = FakeBroker(full=3)

    stats = await publish_rows(make_rows(2), broker.publish)

    first, second = task_id_for(BATCH, 1), task_id_for(BATCH, 2)
    assert [task.task_id for task in broker.attempts] == [first, first, first, first, second]
    assert [task.task_id for task in broker.accepted] == [first, second]
    assert (stats.published, stats.nacked, stats.last_row) == (2, 3, 2)
    assert stats.waited > 0


async def test_confirm_resets_backoff():
    """После ack пауза снова начинается с initial: место появилось — по потолку не спим."""
    backoff = Backoff(initial=0.001, maximum=0.004)

    await publish_rows(make_rows(1), FakeBroker(full=2).publish, backoff)

    # Без сброса следующая пауза была бы 0.002–0.004
    assert backoff.next_delay() <= 0.001


async def test_source_holds_one_unconfirmed_task():
    """Буфера нет: из файла взята ровно одна строка сверх подтверждённых."""
    broker = FakeBroker(full=2)
    pulled = 0
    ahead: list[int] = []

    def counted(rows: list[Row]) -> Iterator[Row]:
        nonlocal pulled
        for row in rows:
            pulled += 1
            yield row

    async def publish(task: TaskMessage) -> None:
        ahead.append(pulled - len(broker.accepted))
        await broker.publish(task)

    await publish_rows(counted(make_rows(5)), publish)

    assert set(ahead) == {1}
    assert len(broker.accepted) == 5


async def test_enqueued_at_is_time_of_accepted_attempt():
    broker = FakeBroker(full=2)

    await publish_rows(make_rows(1), broker.publish)

    assert broker.accepted[0].enqueued_at > broker.attempts[0].enqueued_at


async def test_missing_queue_is_not_retried():
    broker = FakeBroker(error=unroutable())

    with pytest.raises(PublishError):
        await publish_rows(make_rows(3), broker.publish)

    assert len(broker.attempts) == 1


async def test_invalid_urls_are_skipped():
    rows = [Row(1, ""), Row(2, "http://ok.example/"), Row(3, "http://x.example/" + "a" * 3000)]
    broker = FakeBroker()

    stats = await publish_rows(rows, broker.publish)

    assert [task.url for task in broker.accepted] == ["http://ok.example/"]
    assert (stats.published, stats.skipped, stats.last_row) == (1, 2, 3)


async def test_stop_interrupts_backoff_and_keeps_resume_point():
    """Остановка во время паузы: неподтверждённая строка не считается обработанной."""
    accepted: list[TaskMessage] = []

    async def first_accepted_then_full(task: TaskMessage) -> None:
        if accepted:
            raise nack()
        accepted.append(task)

    stop = asyncio.Event()
    stats = SourceStats()
    batch = asyncio.create_task(
        run_batch(make_rows(3), first_accepted_then_full, BATCH, Backoff(10, 10), stop, stats)
    )
    await asyncio.sleep(0.05)
    stop.set()

    assert await asyncio.wait_for(batch, timeout=1) is False
    assert (stats.published, stats.nacked, stats.last_row) == (1, 1, 1)


async def test_stop_after_confirm_does_not_take_next_row():
    """Сигнал пришёл, пока ждали подтверждения: после ack следующая строка не берётся."""
    stop = asyncio.Event()
    broker = FakeBroker()

    async def publish(task: TaskMessage) -> None:
        await broker.publish(task)
        stop.set()

    stats = SourceStats()
    done = await run_batch(make_rows(3), publish, BATCH, Backoff(0.001, 0.004), stop, stats)

    assert done is False
    assert (len(broker.accepted), stats.last_row) == (1, 1)


async def test_progress_report_shows_rate(caplog):
    caplog.set_level(logging.INFO, logger="mlops_phishing.source")
    stats = SourceStats()
    reporter = asyncio.create_task(source.report_progress(stats, interval=0.05))
    await asyncio.sleep(0)  # репортёр запомнил исходные счётчики

    stats.published, stats.nacked = 5, 2
    await asyncio.sleep(0.07)
    reporter.cancel()

    assert "100.0 задач/с, отказов брокера 2" in caplog.text


async def test_publisher_sends_task_with_its_id_to_tasks_queue():
    # delivery_mode TestRabbitBroker не передаёт: persistent проверен на живом брокере
    broker = RabbitBroker()
    received: list[tuple[TaskMessage, RabbitMessage]] = []

    @broker.subscriber(TASKS_QUEUE)
    async def handle(task: TaskMessage, message: RabbitMessage) -> None:
        received.append((task, message))

    task = TaskMessage(
        task_id=task_id_for(BATCH, 1),
        url="http://ok.example/",
        source="batch",
        enqueued_at="2026-10-20T12:00:00Z",
    )
    async with TestRabbitBroker(broker):
        await source.broker_publisher(broker)(task)

    [(delivered, message)] = received
    assert delivered == task
    assert message.message_id == str(task.task_id)


async def test_run_publishes_file_through_confirming_channel(tmp_path, caplog, monkeypatch):
    caplog.set_level(logging.INFO, logger="mlops_phishing.source")
    path = write_csv(tmp_path / "urls.csv", [f"http://site{i}.example/" for i in range(5)])
    broker = RabbitBroker()
    received: list[TaskMessage] = []
    opened: list[dict[str, object]] = []

    @broker.subscriber(TASKS_QUEUE)
    async def handle(task: TaskMessage) -> None:
        received.append(task)

    def open_broker(url: str, **options: object) -> RabbitBroker:
        opened.append(options)
        return broker

    monkeypatch.setattr(source, "RabbitBroker", open_broker)
    args = source.parse_args(
        ["--file", str(path), "--skip", "1", "--limit", "2", "--batch-id", "b1"]
    )

    async with TestRabbitBroker(broker):
        stats = await source.run(args, Settings())

    assert [task.task_id for task in received] == [task_id_for("b1", 2), task_id_for("b1", 3)]
    assert stats.last_row == 3
    # Без on_return_raises публикация в несуществующую очередь молча «удалась» бы
    assert opened == [{"default_channel": PUBLISH_CHANNEL}]
    assert PUBLISH_CHANNEL.publisher_confirms and PUBLISH_CHANNEL.on_return_raises
    assert "Готово: отправлено 2" in caplog.text
    assert "Продолжить" not in caplog.text


async def test_run_stops_on_sigterm_and_logs_resume_point(tmp_path, caplog, monkeypatch):
    """SIGTERM при полной очереди: подтверждённое учтено, продолжение можно скопировать из лога."""
    caplog.set_level(logging.INFO, logger="mlops_phishing.source")
    path = write_csv(tmp_path / "urls.csv", [f"http://site{i}.example/" for i in range(5)])
    accepted: list[TaskMessage] = []

    async def first_accepted_then_sigterm(task: TaskMessage) -> None:
        if accepted:
            signal.raise_signal(signal.SIGTERM)
            raise nack()
        accepted.append(task)

    broker = RabbitBroker()
    monkeypatch.setattr(source, "RabbitBroker", lambda url, **options: broker)
    monkeypatch.setattr(source, "broker_publisher", lambda _: first_accepted_then_sigterm)
    args = source.parse_args(["--file", str(path), "--limit", "4", "--batch-id", "b2"])

    async with TestRabbitBroker(broker):
        stats = await asyncio.wait_for(source.run(args, Settings()), timeout=5)

    assert (stats.published, stats.last_row) == (1, 1)
    assert "Остановлено: отправлено 1" in caplog.text
    assert f"--batch-id b2 --skip 1 --limit 3 --file {path}" in caplog.text


def test_parse_args_defaults_and_rejects_negative():
    args = source.parse_args([])

    assert (args.file, args.limit, args.skip, args.batch_id) == (source.DEFAULT_CSV, None, 0, None)
    with pytest.raises(SystemExit):
        source.parse_args(["--limit", "-1"])


def test_main_requires_existing_file(tmp_path):
    with pytest.raises(SystemExit, match="Нет файла"):
        source.main(["--file", str(tmp_path / "missing.csv")])


def test_main_exits_when_queue_is_missing(tmp_path, monkeypatch):
    path = write_csv(tmp_path / "urls.csv", ["http://a.example/"])
    monkeypatch.setattr(source, "run", mock.AsyncMock(side_effect=unroutable()))

    with pytest.raises(SystemExit) as exit_info:
        source.main(["--file", str(path)])

    assert exit_info.value.code == 1
