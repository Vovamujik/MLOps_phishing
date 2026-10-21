"""Прогон эксперимента: источник через конвейер с ограничениями и без них.

    uv run python -m experiments.run limited
    uv run python -m experiments.run unlimited

Прогон:
1. поднимает стек в нужном режиме с одним воркером и очищает очередь tasks;
2. запускает источник (docker compose run source) на --tasks строк CSV;
3. через --second-worker-at секунд добавляет второй воркер, код и настройки те же;
4. раз в секунду пишет samples.csv: глубина очереди, память брокера и контейнеров;
5. когда все опубликованные задачи записаны в базу, выгружает оттуда tasks.csv:
   когда брокер принял задачу и когда воркер записал результат;
6. возвращает стек в обычный режим.

Скорость источника, пропускная способность и время задачи считаются по tasks.csv
(experiments.analysis), а не по счётчикам RabbitMQ: publish у очереди учитывает и
отвергнутые публикации, а enqueued_at — время именно принятой.

Глубина (ready) и число потребителей — passive declare по AMQP: отвечает сама
очередь, точно и сразу. Management API отдаёт счётчики quorum-очереди раз в 5 с
(тик очереди), поэтому оттуда только unacked и память очереди — с шагом 5 с.
"""

import argparse
import asyncio
import base64
import csv
import json
import os
import re
import shutil
import subprocess
import threading
import time
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote, unquote, urlsplit

import asyncpg
import docker
import docker.errors
import pika

from mlops_phishing.config import Settings, get_settings
from mlops_phishing.messaging import TASKS_QUEUE_NAME

PROJECT = "mlops-phishing"
RESULTS = Path(__file__).parent / "results"

# Одинаковая нагрузка в обоих режимах: 4 потока по 0.2 с — 20 задач/с на воркер
COMMON = {"WORKER_POOL_SIZE": "4", "INFERENCE_DELAY_SECONDS": "0.2"}
MODES = {
    # Ограничения как в обычной работе: лимит очереди и prefetch = пул
    "limited": COMMON | {"TASKS_QUEUE_MAX_LENGTH": "100", "WORKER_PREFETCH": "4"},
    # Без ограничений: очередь растёт без nack, воркер забирает всё, что дадут
    "unlimited": COMMON | {"TASKS_QUEUE_MAX_LENGTH": "0", "WORKER_PREFETCH": "0"},
}
SAMPLE_FIELDS = [
    "t",
    "ready",
    "unacked",
    "consumers",
    "queue_memory",
    "mem_rabbitmq",
    "mem_worker",
    "mem_source",
    "mem_app",
    "workers",
]


def compose(*args: str, env: dict[str, str] | None = None) -> None:
    subprocess.run(["docker", "compose", *args], env=os.environ | (env or {}), check=True)  # noqa: S603, S607


def management(settings: Settings, path: str, method: str = "GET") -> dict[str, Any] | None:
    """Запрос к HTTP API RabbitMQ с логином и паролем из AMQP-адреса."""
    amqp = urlsplit(settings.rabbitmq_url)
    credentials = f"{unquote(amqp.username or '')}:{unquote(amqp.password or '')}"
    request = urllib.request.Request(  # noqa: S310 — адрес из настроек
        f"{settings.rabbitmq_management_url.rstrip('/')}/api/{path}",
        method=method,
        headers={"Authorization": "Basic " + base64.b64encode(credentials.encode()).decode()},
    )
    with urllib.request.urlopen(request, timeout=5) as response:  # noqa: S310
        body = response.read()
    return json.loads(body) if body else None


def queue_path(settings: Settings) -> str:
    vhost = unquote(urlsplit(settings.rabbitmq_url).path.removeprefix("/")) or "/"
    return f"queues/{quote(vhost, safe='')}/{TASKS_QUEUE_NAME}"


def memory_bytes(stats: dict[str, Any]) -> int:
    """Память контейнера, как в docker stats: без файлового кэша, который ядро вытеснит."""
    memory = stats.get("memory_stats", {})
    detail = memory.get("stats", {})
    cache = detail.get("inactive_file", detail.get("total_inactive_file", 0))
    return max(0, memory.get("usage", 0) - cache)


class Sampler(threading.Thread):
    """Раз в секунду: глубина очереди из AMQP и Management API, память контейнеров из Docker."""

    def __init__(self, settings: Settings, started: float, path: Path) -> None:
        super().__init__(daemon=True)
        self.settings = settings
        self.started = started
        self.path = path
        self.docker = docker.from_env()
        self.stopping = threading.Event()
        self.last: dict[str, float] = {}
        self.channel: Any = None

    def sample(self) -> dict[str, float]:
        declared = self.channel.queue_declare(TASKS_QUEUE_NAME, passive=True).method
        queue = management(self.settings, queue_path(self.settings)) or {}
        memory = dict.fromkeys(("rabbitmq", "worker", "source", "app"), 0)
        workers = 0
        label = f"com.docker.compose.project={PROJECT}"
        for container in self.docker.containers.list(filters={"label": label}):
            service = container.labels.get("com.docker.compose.service", "")
            if service in memory:
                memory[service] += memory_bytes(container.stats(stream=False, one_shot=True))
                workers += service == "worker"
        return {
            "t": round(time.monotonic() - self.started, 2),
            "ready": declared.message_count,
            "unacked": queue.get("messages_unacknowledged", 0),
            "consumers": declared.consumer_count,
            "queue_memory": queue.get("memory", 0),
            "workers": workers,
        } | {f"mem_{service}": value for service, value in memory.items()}

    def run(self) -> None:
        # Соединение pika живёт в том потоке, где создано
        connection = pika.BlockingConnection(pika.URLParameters(self.settings.rabbitmq_url))
        self.channel = connection.channel()
        try:
            self.write_samples()
        finally:
            connection.close()

    def write_samples(self) -> None:
        with self.path.open("w", newline="") as file:
            writer = csv.DictWriter(file, SAMPLE_FIELDS)
            writer.writeheader()
            while not self.stopping.wait(1):
                try:
                    self.last = self.sample()
                except (
                    OSError,
                    ValueError,
                    docker.errors.DockerException,
                    pika.exceptions.AMQPError,
                ) as exc:
                    # Брокер или Docker на миг недоступен: пропускаем замер, а не весь прогон
                    print(f"!! замер пропущен: {type(exc).__name__}: {exc}")
                    continue
                writer.writerow(self.last)
                file.flush()


def published_count(log: str) -> int:
    """Сколько задач принял брокер — из итоговой строки источника."""
    found = re.findall(r"Готово: отправлено (\d+)", log)
    if not found:
        raise RuntimeError(f"Источник не дошёл до конца, его лог:\n{log[-2000:]}")
    return int(found[-1])


async def count_done(dsn: str, started_at: datetime) -> int:
    connection = await asyncpg.connect(dsn)
    try:
        return await connection.fetchval(
            "SELECT count(*) FROM tasks"
            " WHERE source = 'batch' AND enqueued_at >= $1 AND status = 'DONE'",
            started_at,
        )
    finally:
        await connection.close()


async def export_tasks(dsn: str, started_at: datetime, path: Path) -> int:
    """Когда брокер принял задачу и когда записан результат — секунды от старта прогона."""
    connection = await asyncpg.connect(dsn)
    try:
        rows = await connection.fetch(
            """
            SELECT enqueued_at, completed_at FROM tasks
            WHERE source = 'batch' AND enqueued_at >= $1 AND status = 'DONE'
            ORDER BY enqueued_at
            """,
            started_at,
        )
    finally:
        await connection.close()
    with path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["enqueued_s", "completed_s"])
        for row in rows:
            writer.writerow(
                [
                    round((row["enqueued_at"] - started_at).total_seconds(), 3),
                    round((row["completed_at"] - started_at).total_seconds(), 3),
                ]
            )
    return len(rows)


async def db_now(dsn: str) -> datetime:
    """Время базы: часы Docker VM, теми же пишутся enqueued_at и completed_at."""
    connection = await asyncpg.connect(dsn)
    try:
        return await connection.fetchval("SELECT now()")
    finally:
        await connection.close()


def run(mode: str, tasks: int, second_worker_at: float, max_seconds: float) -> Path:
    env = MODES[mode]
    settings = get_settings()
    out = RESULTS / mode
    # Пишем во временную папку: прерванный прогон не смешается с прошлым результатом
    partial = RESULTS / f".{mode}.partial"
    shutil.rmtree(partial, ignore_errors=True)
    partial.mkdir(parents=True)
    source_name = f"experiment-source-{mode}"
    client = docker.from_env()

    # Контейнер от прерванного прогона с тем же именем не дал бы запустить источник
    for stale in client.containers.list(all=True, filters={"name": source_name}):
        stale.remove(force=True)

    print(f"== {mode}: поднимаю стек с одним воркером ({env})")
    compose("up", "-d", "--wait", "--scale", "worker=1", env=env)
    # Очистка не трогает выданные воркеру сообщения: после прерванного прогона воркер
    # с prefetch 0 держал бы до 2000 старых задач. Остановленный — вернёт их в очередь
    compose("stop", "worker")
    management(settings, f"{queue_path(settings)}/contents", method="DELETE")
    compose("up", "-d", "--wait", "--scale", "worker=1", env=env)

    started_at = asyncio.run(db_now(settings.database_url))
    started = time.monotonic()
    events: dict[str, float] = {}
    published: int | None = None
    sampler = Sampler(settings, started, partial / "samples.csv")
    sampler.start()
    try:
        batch_id = f"{mode}-{started_at:%Y%m%d-%H%M%S}"
        compose(
            "run", "-d", "--name", source_name, "source",
            "--limit", str(tasks), "--batch-id", batch_id,
            env=env,
        )  # fmt: skip
        events["source_started"] = time.monotonic() - started
        while (elapsed := time.monotonic() - started) < max_seconds:
            if "second_worker" not in events and elapsed >= second_worker_at:
                print(f"== {elapsed:.0f} с: второй воркер")
                compose(
                    "up", "-d", "--no-deps", "--no-recreate", "--scale", "worker=2", "worker",
                    env=env,
                )  # fmt: skip
                events["second_worker"] = elapsed
            if published is None:
                source = client.containers.get(source_name)
                if source.status == "exited":
                    events["source_finished"] = elapsed
                    published = published_count(source.logs().decode())
                    print(f"== {elapsed:.0f} с: источник закончил, принято брокером {published}")
            # Конец — когда все принятые задачи записаны: точнее, чем «очередь пуста» по статистике
            elif asyncio.run(count_done(settings.database_url, started_at)) >= published:
                events["drained"] = elapsed
                break
            time.sleep(1)
        else:
            print(f"!! прогон обрезан по --max-seconds {max_seconds}: результаты неполные")
    finally:
        sampler.stopping.set()
        sampler.join()
        source = client.containers.get(source_name)
        log = source.logs().decode()
        (partial / "source.log").write_text(log)
        source.remove(force=True)

    run_seconds = time.monotonic() - started
    samples_file = partial / "samples.csv"
    samples = len(samples_file.read_text().splitlines()) - 1 if samples_file.exists() else 0
    if samples < 0.8 * run_seconds:
        print(f"!! замеров {samples} за {run_seconds:.0f} с: графики глубины и памяти неполные")

    done = asyncio.run(export_tasks(settings.database_url, started_at, partial / "tasks.csv"))
    nacks = re.findall(r"отказов брокера (\d+), пропущено", log)
    meta = {
        "mode": mode,
        "env": env,
        "tasks": tasks,
        "published": published,
        "done": done,
        "complete": "drained" in events,
        "nacked": int(nacks[-1]) if nacks else None,
        "started_at": started_at.isoformat(),
        "events": {name: round(value, 2) for name, value in events.items()},
    }
    (partial / "run.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    # Готовый прогон заменяет прошлый целиком
    shutil.rmtree(out, ignore_errors=True)
    partial.rename(out)
    print(f"== готово: {done} задач, результаты в {out}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Прогон эксперимента с ограничениями и без.")
    parser.add_argument("mode", choices=sorted(MODES))
    parser.add_argument("--tasks", type=int, default=3000, help="сколько строк CSV отправить")
    parser.add_argument(
        "--second-worker-at", type=float, default=60, help="через сколько секунд второй воркер"
    )
    parser.add_argument("--max-seconds", type=float, default=600, help="предел длительности")
    args = parser.parse_args()
    try:
        run(args.mode, args.tasks, args.second_worker_at, args.max_seconds)
    finally:
        print("== возвращаю обычный режим: лимит и prefetch из .env, один воркер")
        compose("up", "-d", "--wait", "--scale", "worker=1")


if __name__ == "__main__":
    main()
