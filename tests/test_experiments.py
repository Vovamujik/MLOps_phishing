"""Тесты расчётов эксперимента: из этих цифр делаются выводы в README."""

import json
import math

import pytest

from experiments.analysis import (
    SECOND_WORKER_WARMUP,
    WARMUP,
    Run,
    Task,
    backlog,
    cell,
    latency_stats,
    load,
    markdown_table,
    per_second,
    phases,
    rate,
    rolling_mean,
    summary,
)
from experiments.run import memory_bytes, published_count, queue_path
from mlops_phishing.config import Settings


def sample(t: float, ready: float = 0, consumers: float = 1, **memory: float) -> dict[str, float]:
    return {
        "t": t,
        "ready": ready,
        "unacked": 4,
        "consumers": consumers,
        "queue_memory": 0,
        "mem_rabbitmq": memory.get("rabbitmq", 0),
        "mem_worker": memory.get("worker", 0),
        "mem_source": memory.get("source", 0),
        "mem_app": 0,
        "workers": consumers,
    }


def make_run() -> Run:
    """Источник 40 с по 20 задач/с, второй воркер с 20-й секунды, задача идёт 5 с."""
    tasks = [Task(enqueued=i / 20, completed=i / 20 + 5) for i in range(800)]
    samples = [
        sample(t, ready=100, consumers=2 if t >= 20 else 1, rabbitmq=2**20 * t, worker=2**20)
        for t in range(46)
    ]
    meta = {
        "mode": "limited",
        "nacked": 7,
        "events": {
            "source_started": 0.0,
            "second_worker": 20.0,
            "source_finished": 40.0,
            "drained": 48.0,
        },
    }
    return Run("limited", meta, samples, tasks)


def test_per_second_counts_events_into_whole_seconds():
    assert per_second([0.1, 0.9, 1.0, 2.5, 5.0, -1], end=3) == [2, 1, 1]


def test_rolling_mean_smooths_over_window():
    assert rolling_mean([0, 10, 20, 30], window=2) == [0, 5, 15, 25]


def test_rate_is_events_per_second_in_window():
    assert rate([0.5, 1.5, 2.5, 3.5], start=1, end=3) == 1
    assert rate([1, 2], start=5, end=5) == 0


def test_latency_stats_mean_and_p95():
    tasks = [Task(0, float(seconds)) for seconds in range(1, 101)]

    mean, p95 = latency_stats(tasks, 0, 1000)

    assert mean == pytest.approx(50.5)
    assert p95 == pytest.approx(95.95)


def test_latency_stats_without_data_is_nan():
    assert all(math.isnan(value) for value in latency_stats([], 0, 10))
    assert latency_stats([Task(0, 3)], 0, 10) == (3, 3)


def test_phases_split_at_second_worker_and_end_at_last_result():
    """Простой после последнего результата (ждём пустую очередь) не занижает скорость."""
    run = make_run()

    assert run.last_completed == pytest.approx(44.95)
    assert phases(run) == {
        "1 воркер": (WARMUP, 20),
        "2 воркера": (20 + SECOND_WORKER_WARMUP, pytest.approx(44.95)),
    }


def test_summary_rates_follow_phases():
    result = summary(make_run())

    assert result["задач"] == 800
    assert result["отказов брокера"] == 7
    assert result["источник в среднем, задач/с"] == 20
    assert result["1 воркер: источник, задач/с"] == 20
    assert result["2 воркера: источник, задач/с"] == 20
    assert result["1 воркер: время задачи, среднее, с"] == 5
    # Медиана по фазе, а не пик за весь прогон: память RabbitMQ в make_run растёт с t
    assert result["1 воркер: память RabbitMQ, медиана, МиБ"] == 12
    # На один воркер: два воркера по 1 МиБ — это 1 МиБ, а не рост до 2
    assert result["2 воркера: память одного воркера, макс., МиБ"] == 0.5


def test_backlog_is_accepted_minus_written_each_second():
    """Пять задач приняты в первую секунду, записываются по одной в секунду."""
    tasks = [Task(enqueued=0.5, completed=1.5 + i) for i in range(5)]
    run = Run("unlimited", {"events": {"drained": 7.0}}, [], tasks)

    assert backlog(run) == [5, 4, 3, 2, 1, 0, 0]


def test_burst_source_rate_is_not_hidden_by_warmup():
    """Как unlimited: источник выгрузил всё за секунду, ещё до установившегося режима."""
    tasks = [Task(1 + i / 400, 2 + i / 10) for i in range(400)]
    meta = {"mode": "unlimited", "events": {"source_started": 1.0, "drained": 42.0}}

    result = summary(Run("unlimited", meta, [sample(t) for t in range(43)], tasks))

    assert result["источник в среднем, задач/с"] == pytest.approx(400, rel=0.01)
    assert result["макс. задач в очереди (готовые + у воркеров)"] == 400
    assert "1 воркер: источник, задач/с" not in result
    assert result["1 воркер: обработано, задач/с"] == pytest.approx(10, abs=0.3)


def test_second_phase_is_dropped_if_worker_never_settled():
    """Второй воркер запущен за 3 с до конца: установившейся второй фазы нет."""
    run = make_run()
    run.meta["events"]["second_worker"] = 42.0

    assert set(phases(run)) == {"1 воркер"}


def test_missing_values_render_as_dash():
    assert cell(math.nan) == "—"
    assert cell(12.5) == "12.5"


def test_markdown_table_has_column_per_mode():
    table = markdown_table([make_run()])

    assert table.splitlines()[0] == "| Показатель | limited |"
    assert "| задач | 800 |" in table


def test_load_reads_files_written_by_runner(tmp_path):
    (tmp_path / "run.json").write_text(json.dumps({"mode": "unlimited", "events": {}}))
    (tmp_path / "samples.csv").write_text(
        "t,ready,unacked,consumers,queue_memory,mem_rabbitmq,mem_worker,mem_source,mem_app,workers\n"
        "1.0,3000,2000,1,10,100,200,30,40,1\n"
    )
    (tmp_path / "tasks.csv").write_text("enqueued_s,completed_s\n0.5,10.25\n")

    run = load(tmp_path)

    assert run.mode == "unlimited"
    assert run.samples[0]["ready"] == 3000
    assert run.tasks == [Task(0.5, 10.25)]
    assert run.end == 1.0


def test_container_memory_excludes_reclaimable_file_cache():
    """Как docker stats: файловый кэш ядро вытеснит, это не память процесса."""
    cgroup_v2 = {"memory_stats": {"usage": 300, "stats": {"inactive_file": 100}}}
    cgroup_v1 = {"memory_stats": {"usage": 300, "stats": {"total_inactive_file": 50}}}

    assert memory_bytes(cgroup_v2) == 200
    assert memory_bytes(cgroup_v1) == 250
    assert memory_bytes({}) == 0


def test_published_count_comes_from_source_summary():
    log = "Источник: 20.0 задач/с\nГотово: отправлено 3000 (27.6 задач/с), отказов брокера 458"

    assert published_count(log) == 3000
    with pytest.raises(RuntimeError, match="не дошёл до конца"):
        published_count("Прервано ошибкой: отправлено 12 (3.0 задач/с)")


def test_queue_path_encodes_vhost():
    assert queue_path(Settings(rabbitmq_url="amqp://u:p@host:5672/")) == "queues/%2F/tasks"
    assert queue_path(Settings(rabbitmq_url="amqp://u:p@host:5672/lab")) == "queues/lab/tasks"
