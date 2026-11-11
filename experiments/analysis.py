"""Ряды и сводка по прогону эксперимента: только чтение CSV, без matplotlib.

Скорость источника — сколько задач брокер принял за секунду (enqueued_at),
пропускная способность — сколько результатов записано за секунду (completed_at),
время задачи — от приёма брокером до записи результата. Задач в очереди всего
(готовые и уже выданные воркерам) — принятые минус записанные: точно и с шагом
1 с, без Management API, где счётчики quorum-очереди обновляются раз в 5 с.
"""

import csv
import itertools
import json
import math
import statistics
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Переходный режим не в счёт: первые секунды прогона и запуск второго воркера
# (контейнер стартует и грузит модель из MLflow — около 5 с, берём с запасом)
WARMUP = 5.0
SECOND_WORKER_WARMUP = 10.0
MIB = 2**20
KIB = 2**10


@dataclass(frozen=True)
class Task:
    enqueued: float
    completed: float

    @property
    def latency(self) -> float:
        return self.completed - self.enqueued


@dataclass(frozen=True)
class Run:
    mode: str
    meta: dict[str, Any]
    samples: list[dict[str, float]]
    tasks: list[Task]

    def event(self, name: str) -> float | None:
        return self.meta["events"].get(name)

    @property
    def second_worker(self) -> float | None:
        """Когда запущен второй воркер — по журналу прогона, а не по статистике брокера."""
        return self.event("second_worker")

    @property
    def end(self) -> float:
        return self.event("drained") or (self.samples[-1]["t"] if self.samples else 0.0)

    @property
    def last_completed(self) -> float:
        """Последний записанный результат: дальше воркеры уже простаивают."""
        return max((task.completed for task in self.tasks), default=self.end)

    @property
    def source_span(self) -> tuple[float, float]:
        """Первая и последняя задача, принятые брокером: сколько на самом деле работал источник."""
        enqueued = [task.enqueued for task in self.tasks]
        return min(enqueued, default=0.0), max(enqueued, default=0.0)


def load(directory: Path) -> Run:
    meta = json.loads((directory / "run.json").read_text())
    with (directory / "samples.csv").open() as file:
        samples = [
            {key: float(value) for key, value in row.items()} for row in csv.DictReader(file)
        ]
    with (directory / "tasks.csv").open() as file:
        tasks = [
            Task(float(row["enqueued_s"]), float(row["completed_s"]))
            for row in csv.DictReader(file)
        ]
    return Run(meta["mode"], meta, samples, tasks)


def per_second(times: Sequence[float], end: float) -> list[int]:
    """Сколько событий пришлось на каждую секунду отрезка [0, end)."""
    counts = [0] * max(0, math.ceil(end))
    for moment in times:
        if 0 <= moment < len(counts):
            counts[int(moment)] += 1
    return counts


def backlog(run: Run) -> list[int]:
    """Задач в очереди всего на конец каждой секунды: принято брокером минус записано."""
    accepted = per_second([task.enqueued for task in run.tasks], run.end)
    written = per_second([task.completed for task in run.tasks], run.end)
    return list(itertools.accumulate(a - w for a, w in zip(accepted, written, strict=True)))


def rolling_mean(values: Sequence[float], window: int) -> list[float]:
    """Скользящее среднее по последним window точкам: сглаживает пилу от гистерезиса очереди."""
    return [
        statistics.fmean(values[max(0, index - window + 1) : index + 1])
        for index in range(len(values))
    ]


def rate(times: Sequence[float], start: float, end: float) -> float:
    """Событий в секунду на отрезке [start, end)."""
    if end <= start:
        return 0.0
    return sum(start <= moment < end for moment in times) / (end - start)


def latency_stats(tasks: Sequence[Task], start: float, end: float) -> tuple[float, float]:
    """Среднее и 95-й перцентиль времени задач, завершённых на отрезке [start, end)."""
    latencies = [task.latency for task in tasks if start <= task.completed < end]
    if len(latencies) < 2:
        return (latencies[0], latencies[0]) if latencies else (math.nan, math.nan)
    return statistics.fmean(latencies), statistics.quantiles(latencies, n=20)[18]


def phases(run: Run) -> dict[str, tuple[float, float]]:
    """Установившийся режим с одним и с двумя воркерами, без переходных секунд."""
    second, end = run.second_worker, run.last_completed
    if second is None or end <= second + SECOND_WORKER_WARMUP:
        # Второй воркер не вышел на режим до конца прогона — фаза одна
        return {"1 воркер": (WARMUP, min(second or end, end))}
    return {"1 воркер": (WARMUP, second), "2 воркера": (second + SECOND_WORKER_WARMUP, end)}


def summary(run: Run) -> dict[str, object]:
    """Цифры для README: очередь, скорости, время задачи и память по фазам."""
    enqueued = [task.enqueued for task in run.tasks]
    completed = [task.completed for task in run.tasks]
    first, last = run.source_span
    nacked = run.meta.get("nacked")
    complete = run.meta.get("complete", run.event("drained") is not None)
    result: dict[str, object] = {
        "прогон завершён": "да" if complete else "нет",
        "задач": len(run.tasks),
        "отказов брокера": "—" if nacked is None else nacked,
        "партия обработана за, с": round(run.last_completed, 1),
        "источник работал, с": round(last - first, 1),
        "источник в среднем, задач/с": round(len(enqueued) / (last - first), 1)
        if last > first
        else "—",
        "макс. задач в очереди (готовые + у воркеров)": max(backlog(run), default=0),
        "макс. готовых (ready)": max((s["ready"] for s in run.samples), default=0),
        "макс. у воркеров (unacked)": max((s["unacked"] for s in run.samples), default=0),
        "память источника, макс., МиБ": round(
            max((s["mem_source"] for s in run.samples), default=0) / MIB, 1
        ),
    }
    for name, (start, end) in phases(run).items():
        mean, p95 = latency_stats(run.tasks, start, end)
        window = [s for s in run.samples if start <= s["t"] < end] or run.samples
        if start < last:
            # Источник ещё публиковал в этой фазе: иначе его скорость здесь не определена
            result[f"{name}: источник, задач/с"] = round(rate(enqueued, start, min(end, last)), 1)
        result |= {
            f"{name}: обработано, задач/с": round(rate(completed, start, end), 1),
            f"{name}: время задачи, среднее, с": round(mean, 2),
            f"{name}: время задачи, p95, с": round(p95, 2),
            # Медиана, а не пик: память RabbitMQ пилой скачет на сборке мусора
            f"{name}: память RabbitMQ, медиана, МиБ": round(
                statistics.median(s["mem_rabbitmq"] for s in window) / MIB, 1
            ),
            # На один воркер: сумма двух воркеров выглядела бы как рост памяти
            f"{name}: память одного воркера, макс., МиБ": round(
                max(s["mem_worker"] / max(s["workers"], 1) for s in window) / MIB, 1
            ),
            f"{name}: память очереди, макс., КиБ": round(
                max(s["queue_memory"] for s in window) / KIB
            ),
        }
    return result


def cell(value: object) -> str:
    return "—" if isinstance(value, float) and math.isnan(value) else str(value)


def markdown_table(runs: Sequence[Run]) -> str:
    summaries = [summary(run) for run in runs]
    keys = list(dict.fromkeys(key for item in summaries for key in item))
    lines = [
        "| Показатель | " + " | ".join(run.mode for run in runs) + " |",
        "|---|" + "---|" * len(runs),
    ]
    lines += [
        f"| {key} | " + " | ".join(cell(item.get(key, "—")) for item in summaries) + " |"
        for key in keys
    ]
    return "\n".join(lines)
