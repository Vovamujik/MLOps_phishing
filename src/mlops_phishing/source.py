"""Источник задач: пакетный скоринг большого CSV через ограниченную очередь.

Противодавление «воркер → источник». Очередь tasks ограничена policy
(max-length, overflow=reject-publish), источник публикует с publisher confirms:
- место в очереди есть — брокер отвечает basic.ack, источник берёт следующую строку;
- очередь полна — брокер отвечает basic.nack (aio-pika поднимает DeliveryError),
  источник ждёт и отправляет ту же задачу снова. Пауза растёт экспоненциально,
  с джиттером и потолком: без busy loop и без долгого сна, когда место уже есть.

Буфера нет: в памяти одна неподтверждённая задача, остальные лежат в файле,
который читается построчно. В установившемся режиме ack приходят с той
скоростью, с которой воркеры освобождают место, поэтому скорость источника сама
становится равна их общей пропускной способности. Второй воркер освобождает
место вдвое чаще, и источник ускоряется вдвое без правки кода и настроек.

Почему сигнал — nack от брокера, а не другое:
- опрос глубины через Management API: статистика обновляется раз в 5 с, и несколько
  издателей, увидев «место есть», публикуют разом и переполняют очередь;
- фиксированная частота — это rate limiting: под число воркеров она не подстраивается;
- буфер в памяти источника растёт без предела, при падении задачи в нём теряются;
- memory alarm брокера (connection.blocked) блокирует всех издателей, включая API;
- overflow=drop-head или публикация без confirms: лишние задачи молча теряются.
Пауза после nack — не rate limiting: без отказов источник публикует со скоростью
подтверждений, темп задают ack брокера.

Трейсы: у каждой задачи свой корневой трейс source.task, а не один на всю партию —
трейс на тысячи спанов не открыть и не найти в нём задачу. Внутри — publish каждой
попытки (FastStream пишет traceparent в заголовки, воркер продолжает этот трейс),
отказ брокера — событие broker.rejected, пауза перед повтором — дочерний спан
source.backoff с номером попытки и длительностью.

task_id детерминирован (партия + номер строки): повторный запуск партии,
например после падения, не задублирует результаты — воркер пишет их идемпотентно.

Запуск: python -m mlops_phishing.source --limit 2000
"""

import argparse
import asyncio
import csv
import logging
import random
import signal
import time
import uuid
from collections.abc import Awaitable, Callable, Iterable, Iterator
from contextlib import AbstractContextManager
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from itertools import islice
from pathlib import Path

import structlog
from aio_pika.exceptions import DeliveryError, PublishError
from faststream.rabbit import RabbitBroker
from opentelemetry import context as otel_context
from opentelemetry import trace
from pydantic import ValidationError

from mlops_phishing import instruments, logging_config, telemetry
from mlops_phishing.config import Settings, get_settings
from mlops_phishing.messaging import (
    BROKER_LOGGER,
    PUBLISH_CHANNEL,
    TASKS_QUEUE,
    TaskMessage,
    publish_task,
    telemetry_middleware,
)

# Не __name__: под python -m он равен __main__
log = logging.getLogger("mlops_phishing.source")

DEFAULT_CSV = Path("data/raw/PhiUSIIL_Phishing_URL_Dataset.csv")
URL_COLUMN = "URL"
# Пространство имён uuid5: task_id для пары (партия, строка) одинаков при каждом запуске
TASK_NAMESPACE = uuid.UUID("6f1d2c4e-9a51-4b8e-8f0a-3c7d5e2b1a90")
# Как часто писать в лог скорость источника, секунды
REPORT_EVERY = 5.0

Publish = Callable[[TaskMessage], Awaitable[None]]


@dataclass(frozen=True)
class Row:
    number: int  # номер строки данных с 1, без заголовка
    url: str


@dataclass
class SourceStats:
    published: int = 0  # подтверждено брокером
    nacked: int = 0  # отказов брокера: очередь была полна
    skipped: int = 0  # строки с некорректным URL
    waited: float = 0.0  # сумма пауз после отказов, секунды
    last_row: int = 0  # последняя обработанная строка: с неё продолжается партия


class Backoff:
    """Экспоненциальная пауза с джиттером: 0.05, 0.1, 0.2 … до потолка."""

    def __init__(self, initial: float, maximum: float) -> None:
        self.initial = min(initial, maximum)
        self.maximum = maximum
        self._next = self.initial

    def next_delay(self) -> float:
        delay = self._next
        self._next = min(self.maximum, delay * 2)
        # Половина паузы гарантирована, повтора без задержки не бывает. Случайная половина
        # разводит по времени несколько источников, чтобы они не стучались в брокер разом
        return delay / 2 + random.uniform(0, delay / 2)  # noqa: S311 — джиттер, не криптография

    def reset(self) -> None:
        self._next = self.initial


def read_urls(path: Path) -> Iterator[Row]:
    """Строки CSV по одной: файл в память целиком не читается."""
    # utf-8-sig: CSV PhiUSIIL начинается с BOM, без него первая колонка не нашлась бы по имени
    with path.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        if URL_COLUMN not in (reader.fieldnames or []):
            raise ValueError(f"В {path} нет колонки {URL_COLUMN}")
        for number, record in enumerate(reader, start=1):
            yield Row(number, record[URL_COLUMN])


def task_id_for(batch_id: str, row: int) -> uuid.UUID:
    return uuid.uuid5(TASK_NAMESPACE, f"{batch_id}:{row}")


async def wait_or_stop(stop: asyncio.Event, delay: float) -> bool:
    """Пауза, которую прерывает остановка. True — пора останавливаться."""
    try:
        await asyncio.wait_for(stop.wait(), timeout=delay)
    except TimeoutError:
        return False
    return True


async def publish_confirmed(
    task: TaskMessage,
    publish: Publish,
    backoff: Backoff,
    stats: SourceStats,
    stop: asyncio.Event,
) -> bool:
    """Отправлять задачу, пока брокер её не подтвердит. False — прервано остановкой.

    Отказ брокера — штатное противодавление, а не ошибка. Первый отказ серии пишется
    на INFO (видно, когда источник упёрся в лимит и на сколько замедлился), остальные —
    на DEBUG: на каждый nack было бы 5–20 строк в секунду. Темп виден в отчёте раз в 5 с.
    """
    span = trace.get_current_span()
    attempt = 0
    while True:
        attempt += 1
        try:
            # enqueued_at — момент попытки: время в очереди считается от реальной постановки
            await publish(task.model_copy(update={"enqueued_at": datetime.now(UTC)}))
        except PublishError:
            # Брокеру некуда доставить сообщение (нет очереди): ожиданием это не лечится
            raise
        except DeliveryError:
            stats.nacked += 1
            delay = backoff.next_delay()
            stats.waited += delay
            instruments.source_rejections.add(1)
            instruments.source_backoff.add(delay)
            log.log(
                logging.INFO if attempt == 1 else logging.DEBUG,
                "Очередь заполнена, брокер отказал: пауза %.2f с перед попыткой %d",
                delay,
                attempt + 1,
                extra={"attempt": attempt, "delay": round(delay, 3)},
            )
            span.add_event("broker.rejected", {"attempt": attempt, "delay_s": delay})
            if await backoff_wait(stop, delay, attempt):
                span.add_event("stopped")
                return False
        else:
            backoff.reset()
            stats.published += 1
            instruments.source_published.add(1)
            span.set_attribute("source.attempts", attempt)
            return True


async def backoff_wait(stop: asyncio.Event, delay: float, attempt: int) -> bool:
    """Пауза после отказа — отдельный спан: на таймлайне видно, сколько задача ждала."""
    attributes = {"attempt": attempt, "delay_s": delay}
    with telemetry.tracer().start_as_current_span("source.backoff", attributes=attributes):
        return await wait_or_stop(stop, delay)


async def run_batch(
    rows: Iterable[Row],
    publish: Publish,
    batch_id: str,
    backoff: Backoff,
    stop: asyncio.Event,
    stats: SourceStats,
) -> bool:
    """Публиковать строки по одной, следующую — только после подтверждения предыдущей.

    True — строки кончились, False — остановлено сигналом.
    """
    for row in rows:
        if stop.is_set():
            return False
        task_id = task_id_for(batch_id, row.number)
        # task_id и номер строки — во всех записях о строке; сам URL в лог не пишем
        with (
            structlog.contextvars.bound_contextvars(task_id=str(task_id), row=row.number),
            task_span(task_id, batch_id, row.number),
        ):
            if not await publish_row(row, task_id, publish, backoff, stats, stop):
                return False
        stats.last_row = row.number
    return True


def task_span(task_id: uuid.UUID, batch_id: str, row: int) -> AbstractContextManager[trace.Span]:
    """Корневой спан задачи: пустой родительский контекст — новый трейс на каждую строку."""
    return telemetry.tracer().start_as_current_span(
        "source.task",
        context=otel_context.Context(),
        kind=trace.SpanKind.PRODUCER,
        attributes={"task.id": str(task_id), "source.batch_id": batch_id, "source.row": row},
    )


async def publish_row(
    row: Row,
    task_id: uuid.UUID,
    publish: Publish,
    backoff: Backoff,
    stats: SourceStats,
    stop: asyncio.Event,
) -> bool:
    """Одна строка: проверить URL и отправить. False — прервано остановкой."""
    try:
        task = TaskMessage(
            task_id=task_id, url=row.url, source="batch", enqueued_at=datetime.now(UTC)
        )
    except ValidationError:
        stats.skipped += 1
        trace.get_current_span().add_event("row.skipped")
        log.warning("Строка %d пропущена: URL пустой или длиннее лимита", row.number)
        return True
    return await publish_confirmed(task, publish, backoff, stats, stop)


async def report_progress(stats: SourceStats, interval: float = REPORT_EVERY) -> None:
    """Скорость источника раз в interval секунд: видно, как она подстраивается под воркеры."""
    published, nacked = stats.published, stats.nacked
    while True:
        await asyncio.sleep(interval)
        log.info(
            "Источник: %.1f задач/с, отказов брокера %d за %.0f с, всего отправлено %d",
            (stats.published - published) / interval,
            stats.nacked - nacked,
            interval,
            stats.published,
        )
        published, nacked = stats.published, stats.nacked


def broker_publisher(broker: RabbitBroker) -> Publish:
    # Без таймаута: если брокер завис (memory alarm), источнику и надо ждать — это тоже давление
    async def publish(task: TaskMessage) -> None:
        await publish_task(broker, task)

    return publish


def request_stop(stop: asyncio.Event, signum: signal.Signals) -> None:
    """Начало graceful shutdown: текущая задача дождётся ответа брокера, новых не будет."""
    log.info("Получен %s: останавливаемся после текущей задачи", signum.name)
    stop.set()


async def run(args: argparse.Namespace, settings: Settings) -> SourceStats:
    batch_id = args.batch_id or uuid.uuid4().hex[:12]
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for signum in (signal.SIGINT, signal.SIGTERM):
        # Остановка между задачами: текущая дождётся ответа брокера, в памяти ничего не останется
        loop.add_signal_handler(signum, request_stop, stop, signum)

    stats = SourceStats(last_row=args.skip)
    end = None if args.limit is None else args.skip + args.limit
    rows = islice(read_urls(args.file), args.skip, end)
    backoff = Backoff(settings.source_backoff_initial, settings.source_backoff_max)
    log.info(
        "Партия %s: %s начиная со строки %d",
        batch_id,
        args.file,
        args.skip + 1,
        extra={"batch_id": batch_id},
    )

    started = time.monotonic()
    reporter = asyncio.create_task(report_progress(stats))
    status = "Прервано ошибкой"
    try:
        async with RabbitBroker(
            settings.rabbitmq_url,
            default_channel=PUBLISH_CHANNEL,
            logger=BROKER_LOGGER,
            middlewares=[telemetry_middleware()],
        ) as broker:
            publish = broker_publisher(broker)
            done = await run_batch(rows, publish, batch_id, backoff, stop, stats)
        status = "Готово" if done else "Остановлено"
    finally:
        reporter.cancel()
        elapsed = time.monotonic() - started
        resume = ""
        if status != "Готово":
            resume = f". Продолжить партию: --batch-id {batch_id} --skip {stats.last_row}"
            if end is not None:
                resume += f" --limit {end - stats.last_row}"
            if args.file != DEFAULT_CSV:
                resume += f" --file {args.file}"
        log.info(
            "%s: отправлено %d (%.1f задач/с), отказов брокера %d, пропущено %d, в паузах %.1f с%s",
            status,
            stats.published,
            stats.published / elapsed if elapsed else 0.0,
            stats.nacked,
            stats.skipped,
            stats.waited,
            resume,
            extra={"batch_id": batch_id, **asdict(stats)},
        )
    return stats


def non_negative(value: str) -> int:
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("нужно число не меньше 0")
    return number


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Источник задач: URL из CSV в очередь tasks.")
    parser.add_argument("--file", type=Path, default=DEFAULT_CSV, help="CSV с колонкой URL")
    parser.add_argument("--limit", type=non_negative, help="сколько строк отправить (все)")
    parser.add_argument("--skip", type=non_negative, default=0, help="пропустить первые строки (0)")
    parser.add_argument(
        "--batch-id", help="id партии, чтобы продолжить ту же партию того же файла (новый)"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if not args.file.is_file():
        raise SystemExit(f"Нет файла {args.file}: uv run python -m training.download_data")
    settings = get_settings()
    logging_config.setup_logging(settings.log_level, "source", settings.environment)
    tracing = telemetry.setup_telemetry("source")
    try:
        asyncio.run(run(args, settings))
    except PublishError:
        log.error("Очередь %s не существует: сначала поднимите rabbitmq-init", TASKS_QUEUE.name)
        raise SystemExit(1) from None
    finally:
        # Процесс короткий: без явного сброса последние спаны и срез метрик (экспорт
        # раз в 5 с) остались бы в памяти, и короткая партия не попала бы на дашборд
        tracing.shutdown()


if __name__ == "__main__":
    main()
