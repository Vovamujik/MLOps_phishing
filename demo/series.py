"""Сколько временных рядов у метрик приложения сейчас и сколько станет с лейблом task_id.

Число рядов метрики — произведение числа значений её лейблов. Задача касается одного
воркера и одного результата, поэтому лейбл task_id умножает ряды «на одну задачу»
(у гистограммы — бакеты + sum + count, у счётчика — 1) на число задач в файле:
после 6000 строк одна гистограмма inference — 6000 × 17 ≈ 102 тыс. рядов.

    uv run python -m demo.series --rows 6000
"""

import argparse
import json
import urllib.parse
import urllib.request
from dataclasses import dataclass

# Метрики, которые пишутся на каждую задачу: к ним и добавился бы task_id
PER_TASK = r"worker_.*|source_.*|http_server_request_duration_seconds.*"
APP_JOBS = "api|worker|source"
HISTOGRAM_SUFFIXES = ("_bucket", "_sum", "_count")


@dataclass(frozen=True)
class Family:
    name: str
    series: int  # рядов сейчас
    per_label_set: int  # рядов на один набор лейблов: бакеты + sum + count или 1

    def with_task_id(self, rows: int) -> int:
        return rows * self.per_label_set


def family_name(metric: str) -> str:
    for suffix in HISTOGRAM_SUFFIXES:
        if metric.endswith(suffix):
            return metric.removesuffix(suffix)
    return metric


def group_families(counts: dict[str, int], label_sets: dict[str, int]) -> list[Family]:
    """Ряды по метрикам → семейства: X_bucket, X_sum и X_count — одна гистограмма.

    label_sets — сколько разных наборов лейблов (без le) у X_bucket: делением получаем
    число бакетов в одном наборе.
    """
    totals: dict[str, int] = {}
    for metric, count in counts.items():
        totals[family_name(metric)] = totals.get(family_name(metric), 0) + count
    families = []
    for name, total in sorted(totals.items()):
        bucket = counts.get(f"{name}_bucket")
        # У гистограммы на набор лейблов — его бакеты + sum + count, у остальных — 1 ряд
        sets = label_sets.get(name)
        per_set = bucket // sets + 2 if bucket and sets else 1
        families.append(Family(name, total, per_set))
    return families


def spaced(number: int) -> str:
    """6000 → «6 000»: разряды через пробел, как в README."""
    return f"{number:_}".replace("_", " ")


def report(families: list[Family], rows: int) -> list[str]:
    lines = [f"{'метрика':40} {'сейчас':>7} {'на задачу':>10} {f'с task_id ({rows} строк)':>24}"]
    for family in families:
        lines.append(
            f"{family.name:40} {family.series:>7} {family.per_label_set:>10} "
            f"{spaced(family.with_task_id(rows)):>24}"
        )
    now = sum(f.series for f in families)
    added = sum(f.with_task_id(rows) for f in families)
    lines.append(f"Итого рядов сейчас: {now}; с лейблом task_id добавится ≈ {spaced(added)}")
    return lines


def query(prometheus: str, promql: str) -> list[dict]:
    url = f"{prometheus}/api/v1/query?" + urllib.parse.urlencode({"query": promql})
    with urllib.request.urlopen(url, timeout=10) as response:  # noqa: S310 — адрес из аргументов
        return json.load(response)["data"]["result"]


def fetch(prometheus: str) -> tuple[dict[str, int], dict[str, int]]:
    selector = f'{{__name__=~"{PER_TASK}", job=~"{APP_JOBS}"}}'
    counts = {
        r["metric"]["__name__"]: int(r["value"][1])
        for r in query(prometheus, f"count by (__name__) ({selector})")
    }
    label_sets = {}
    for metric in counts:
        if metric.endswith("_bucket"):
            result = query(prometheus, f"count(count without (le) ({metric}))")
            label_sets[family_name(metric)] = int(result[0]["value"][1]) if result else 0
    return counts, label_sets


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Оценка числа рядов с лейблом task_id.")
    parser.add_argument("--prometheus", default="http://localhost:9090")
    parser.add_argument("--rows", type=int, default=6000, help="строк в файле (6000)")
    args = parser.parse_args(argv)
    counts, label_sets = fetch(args.prometheus)
    print("\n".join(report(group_families(counts, label_sets), args.rows)))


if __name__ == "__main__":
    main()
