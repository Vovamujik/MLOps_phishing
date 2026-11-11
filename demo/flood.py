"""Поток POST /api/v1/tasks: при полной очереди API сбрасывает нагрузку ответом 503.

Печатает, сколько каких ответов пришло, Retry-After у 503 и X-Request-ID нескольких
отказов: по request_id запись находится в Loki, в ней trace_id — переход к трейсу
ответа 503 (статус ERROR, DeliveryError на спане publish). Чтобы 503 были, очередь
должна быть полна: параллельно идёт источник или воркер остановлен.

    uv run python -m demo.flood --requests 300 --concurrency 20
"""

import argparse
import asyncio
import time
from collections import Counter
from dataclasses import dataclass, field

import httpx

REQUEST_ID_HEADER = "X-Request-ID"
# Сколько request_id отказов показать: больше одного-двух для демонстрации не нужно
SHOWN_REJECTIONS = 3


@dataclass
class FloodResult:
    statuses: Counter[int] = field(default_factory=Counter)
    retry_after: Counter[str] = field(default_factory=Counter)
    rejected_request_ids: list[str] = field(default_factory=list)
    errors: int = 0  # соединение не удалось: API лежит или не успел ответить
    elapsed: float = 0.0

    def record(self, response: httpx.Response) -> None:
        self.statuses[response.status_code] += 1
        if response.status_code == 503:
            self.retry_after[response.headers.get("Retry-After", "нет")] += 1
            request_id = response.headers.get(REQUEST_ID_HEADER)
            if request_id and len(self.rejected_request_ids) < SHOWN_REJECTIONS:
                self.rejected_request_ids.append(request_id)

    def report(self) -> list[str]:
        total = sum(self.statuses.values()) + self.errors
        lines = [f"Запросов {total} за {self.elapsed:.1f} с"]
        lines += [f"  {code}: {count}" for code, count in sorted(self.statuses.items())]
        if self.errors:
            lines.append(f"  без ответа: {self.errors}")
        if self.retry_after:
            values = ", ".join(f"{value} с × {n}" for value, n in self.retry_after.items())
            lines.append(f"Retry-After у 503: {values}")
        for request_id in self.rejected_request_ids:
            # request_id — поле JSON, а не structured metadata: ищем фильтром по строке
            lines.append(f'Loki: {{service="api"}} |= "{request_id}"  → trace_id → Tempo')
        lines.append("Tempo: {span.http.response.status_code=503} (поиск отстаёт на секунды)")
        lines.append("Дашборд: «Доля ответов 503 у POST /api/v1/tasks»")
        return lines


async def flood(
    api_url: str,
    requests: int,
    concurrency: int,
    transport: httpx.AsyncBaseTransport | None = None,
) -> FloodResult:
    """Отправить requests запросов, не больше concurrency одновременно.

    transport — для тестов: подменяет сеть ответами-заглушками.
    """
    result = FloodResult()
    limit = asyncio.Semaphore(concurrency)

    async def one(client: httpx.AsyncClient, number: int) -> None:
        async with limit:
            try:
                response = await client.post(
                    "/api/v1/tasks", json={"url": f"http://flood.example/{number}"}
                )
            except httpx.HTTPError:
                result.errors += 1
            else:
                result.record(response)

    started = time.perf_counter()
    async with httpx.AsyncClient(base_url=api_url, timeout=5.0, transport=transport) as client:
        await asyncio.gather(*(one(client, n) for n in range(requests)))
    result.elapsed = time.perf_counter() - started
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Поток запросов POST /api/v1/tasks.")
    parser.add_argument("--api-url", default="http://localhost:8000")
    parser.add_argument("--requests", type=int, default=300)
    parser.add_argument("--concurrency", type=int, default=20)
    args = parser.parse_args(argv)
    result = asyncio.run(flood(args.api_url, args.requests, args.concurrency))
    print("\n".join(result.report()))


if __name__ == "__main__":
    main()
