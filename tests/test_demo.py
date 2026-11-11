"""Сценарии защиты: логика скриптов demo без брокера, API и Prometheus."""

import json

import httpx
import pytest
from pydantic import ValidationError

from demo import flood, poison, series
from mlops_phishing.messaging import TaskMessage


def test_poison_body_is_rejected_by_worker_schema():
    # Воркер валидирует тело как TaskMessage: битое сообщение должно не пройти
    with pytest.raises(ValidationError):
        TaskMessage.model_validate(json.loads(poison.poison_body()))


def test_poison_id_is_unique_and_searchable():
    first, second = poison.poison_id(), poison.poison_id()
    assert first != second
    assert first.startswith("poison-")
    hints = "\n".join(poison.where_to_look(first))
    assert f'task_id="{first}"' in hints
    assert f'span.task.id="{first}"' in hints


async def test_flood_counts_statuses_and_rejected_request_ids():
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls <= 2:
            return httpx.Response(202, json={})
        return httpx.Response(
            503, headers={"Retry-After": "1", "X-Request-ID": f"req{calls}"}, json={}
        )

    result = await flood.flood(
        "http://api", requests=7, concurrency=3, transport=httpx.MockTransport(handler)
    )

    assert result.statuses == {202: 2, 503: 5}
    assert result.retry_after == {"1": 5}
    # Показываем лишь несколько отказов — по ним идут в Loki и Tempo
    assert len(result.rejected_request_ids) == flood.SHOWN_REJECTIONS
    report = "\n".join(result.report())
    assert "503: 5" in report
    assert "Retry-After у 503: 1 с × 5" in report
    assert '|= "req3"' in report


async def test_flood_counts_connection_errors():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("API лежит")

    result = await flood.flood(
        "http://api", requests=4, concurrency=2, transport=httpx.MockTransport(handler)
    )

    assert result.errors == 4
    assert "без ответа: 4" in "\n".join(result.report())


def test_series_groups_histogram_parts_into_one_family():
    counts = {
        "worker_inference_duration_seconds_bucket": 30,  # 2 воркера × 15 бакетов
        "worker_inference_duration_seconds_sum": 2,
        "worker_inference_duration_seconds_count": 2,
        "worker_tasks_processed_total": 6,
    }
    families = series.group_families(counts, {"worker_inference_duration_seconds": 2})

    by_name = {f.name: f for f in families}
    inference = by_name["worker_inference_duration_seconds"]
    assert inference.series == 34
    assert inference.per_label_set == 17  # 15 бакетов + sum + count
    assert inference.with_task_id(6000) == 102_000
    assert by_name["worker_tasks_processed_total"].per_label_set == 1


def test_series_report_sums_and_formats_numbers():
    families = [series.Family("a", 34, 17), series.Family("b", 6, 1)]

    lines = series.report(families, rows=6000)

    assert "102 000" in lines[1]
    assert lines[-1] == "Итого рядов сейчас: 40; с лейблом task_id добавится ≈ 108 000"
