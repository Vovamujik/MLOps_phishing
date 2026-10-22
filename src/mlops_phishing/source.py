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
from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import islice
from pathlib import Path

from aio_pika.exceptions import DeliveryError, PublishError
from faststream.rabbit import RabbitBroker
from pydantic import ValidationError

from mlops_phishing import logging_config
from mlops_phishing.config import Settings, get_settings
from mlops_phishing.messaging import PUBLISH_CHANNEL, TASKS_QUEUE, TaskMessage, publish_task

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
    """Отправлять задачу, пока брокер её не подтвердит. False — прервано остановкой."""
    while True:
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
            if await wait_or_stop(stop, delay):
                return False
        else:
            backoff.reset()
            stats.published += 1
            return True


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
        try:
            task = TaskMessage(
                task_id=task_id_for(batch_id, row.number),
                url=row.url,
                source="batch",
                enqueued_at=datetime.now(UTC),
            )
        except ValidationError:
            stats.skipped += 1
            log.warning("Строка %d пропущена: URL пустой или длиннее лимита", row.number)
        else:
            if not await publish_confirmed(task, publish, backoff, stats, stop):
                return False
        stats.last_row = row.number
    return True


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


async def run(args: argparse.Namespace, settings: Settings) -> SourceStats:
    batch_id = args.batch_id or uuid.uuid4().hex[:12]
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for signum in (signal.SIGINT, signal.SIGTERM):
        # Остановка между задачами: текущая дождётся ответа брокера, в памяти ничего не останется
        loop.add_signal_handler(signum, stop.set)

    stats = SourceStats(last_row=args.skip)
    end = None if args.limit is None else args.skip + args.limit
    rows = islice(read_urls(args.file), args.skip, end)
    backoff = Backoff(settings.source_backoff_initial, settings.source_backoff_max)
    log.info("Партия %s: %s начиная со строки %d", batch_id, args.file, args.skip + 1)

    started = time.monotonic()
    reporter = asyncio.create_task(report_progress(stats))
    status = "Прервано ошибкой"
    try:
        async with RabbitBroker(settings.rabbitmq_url, default_channel=PUBLISH_CHANNEL) as broker:
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
    logging_config.setup_logging(settings.log_level)
    try:
        asyncio.run(run(args, settings))
    except PublishError:
        log.error("Очередь %s не существует: сначала поднимите rabbitmq-init", TASKS_QUEUE.name)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
