"""Графики и сводная таблица по результатам experiments.run.

    uv run python -m experiments.plot

Нужна группа ml (matplotlib): uv sync --all-groups. Картинки и summary.md
ложатся в experiments/results, рядом с данными прогонов.
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from experiments.analysis import (
    MIB,
    SECOND_WORKER_WARMUP,
    Run,
    backlog,
    load,
    markdown_table,
    per_second,
    rolling_mean,
)

RESULTS = Path(__file__).parent / "results"
MODES = {"limited": "с ограничениями", "unlimited": "без ограничений"}
COLORS = {"limited": "tab:blue", "unlimited": "tab:red"}
QUEUE_LIMIT = 100
SMOOTH_SECONDS = 5


def mark_second_worker(ax: Axes, run: Run) -> None:
    """Пунктир — запуск второго воркера, тень — его разгон, который сводка не учитывает."""
    if (moment := run.second_worker) is None:
        return
    ax.axvline(
        moment,
        color=COLORS[run.mode],
        linestyle="--",
        alpha=0.6,
        label=f"{MODES[run.mode]}: второй воркер",
    )
    ax.axvspan(moment, moment + SECOND_WORKER_WARMUP, color=COLORS[run.mode], alpha=0.08)


def symlog_axis(ax: Axes, threshold: float, ticks: list[int]) -> None:
    """Линейно до threshold, дальше логарифм: видны и лимит 100, и тысячи без лимита."""
    ax.set_yscale("symlog", linthresh=threshold)
    ax.set_yticks(ticks, labels=[str(tick) for tick in ticks])
    ax.set_ylim(bottom=0)


def legend_outside(ax: Axes) -> None:
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize="small")


def plot_depth(runs: list[Run], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 4.5))
    for run in runs:
        color = COLORS[run.mode]
        total = backlog(run)
        ax.step(range(len(total)), total, where="post", color=color, label=MODES[run.mode])
        ax.plot(
            [s["t"] for s in run.samples],
            [s["ready"] for s in run.samples],
            color=color,
            linewidth=0.8,
            alpha=0.5,
            label=f"{MODES[run.mode]}: из них готовых (ready)",
        )
        mark_second_worker(ax, run)
    ax.axhline(QUEUE_LIMIT, color="gray", linewidth=0.8, label=f"лимит {QUEUE_LIMIT}")
    symlog_axis(ax, QUEUE_LIMIT, [0, 50, 100, 300, 1000, 3000])
    ax.set(
        xlabel="секунды",
        ylabel="задач",
        title="Задач в очереди: принято брокером, но ещё не записано",
    )
    legend_outside(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=120)


def plot_memory(runs: list[Run], path: Path) -> None:
    panels = {
        "RabbitMQ": lambda s: s["mem_rabbitmq"],
        "один воркер": lambda s: s["mem_worker"] / max(s["workers"], 1),
        "источник": lambda s: s["mem_source"],
    }
    fig, axes = plt.subplots(1, len(panels), figsize=(15, 4))
    for ax, (title, value) in zip(axes, panels.items(), strict=True):
        for run in runs:
            ax.plot(
                [s["t"] for s in run.samples],
                [value(s) / MIB for s in run.samples],
                color=COLORS[run.mode],
                label=MODES[run.mode],
            )
            mark_second_worker(ax, run)
        ax.set(xlabel="секунды", ylabel="МиБ", title=f"Память: {title}")
        ax.set_ylim(bottom=0)
    legend_outside(axes[-1])
    fig.tight_layout()
    fig.savefig(path, dpi=120)


def plot_rates(runs: list[Run], path: Path) -> None:
    fig, axes = plt.subplots(len(runs), 1, figsize=(11, 3.5 * len(runs)), squeeze=False)
    for ax, run in zip(axes[:, 0], runs, strict=True):
        enqueued = per_second([task.enqueued for task in run.tasks], run.end)
        completed = per_second([task.completed for task in run.tasks], run.end)
        seconds = range(len(enqueued))
        ax.plot(seconds, rolling_mean(enqueued, SMOOTH_SECONDS), label="источник: принято брокером")
        ax.plot(seconds, rolling_mean(completed, SMOOTH_SECONDS), label="воркеры: записано")
        mark_second_worker(ax, run)
        if max(enqueued, default=0) > QUEUE_LIMIT:
            # Всплеск источника в десятки раз выше скорости воркеров
            symlog_axis(ax, 50, [0, 20, 40, 100, 300, 1000])
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.3)
        ax.set(
            xlabel="секунды",
            ylabel=f"задач/с, среднее за {SMOOTH_SECONDS} с",
            title=f"Скорость: {MODES[run.mode]}",
        )
        legend_outside(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=120)


def plot_latency(runs: list[Run], path: Path) -> None:
    fig, axes = plt.subplots(
        len(runs), 1, figsize=(11, 3.5 * len(runs)), squeeze=False, sharex=True
    )
    for ax, run in zip(axes[:, 0], runs, strict=True):
        ax.scatter(
            [task.completed for task in run.tasks],
            [task.latency for task in run.tasks],
            s=2,
            color=COLORS[run.mode],
        )
        mark_second_worker(ax, run)
        ax.set_ylim(bottom=0)
        ax.set(
            ylabel="секунды от приёма до результата",
            title=f"Время выполнения задачи: {MODES[run.mode]}",
        )
    axes[-1, 0].set_xlabel("секунды (когда записан результат)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)


def main() -> None:
    runs = [load(RESULTS / mode) for mode in MODES if (RESULTS / mode / "run.json").exists()]
    if not runs:
        raise SystemExit("Нет прогонов: uv run python -m experiments.run limited")
    plot_depth(runs, RESULTS / "depth.png")
    plot_memory(runs, RESULTS / "memory.png")
    plot_rates(runs, RESULTS / "rates.png")
    plot_latency(runs, RESULTS / "latency.png")
    table = markdown_table(runs)
    (RESULTS / "summary.md").write_text(table + "\n")
    print(table)
    print(f"\nГрафики и summary.md: {RESULTS}")


if __name__ == "__main__":
    main()
