"""«Ядовитое» сообщение: задача, которую воркер не сможет разобрать.

Публикуется прямо в очередь tasks, мимо API: API такое сообщение не пропустил бы
(валидация TaskCreate). Воркер не проходит валидацию TaskMessage, отклоняет его без
возврата, и брокер перекладывает его в tasks.dlq с первой попытки. След:
ERROR «Ошибка обработки, задача переведена в DLQ без повторов» со стектрейсом и
trace_id в Loki, спан process со статусом ERROR и событием dead_lettered в Tempo,
рост DLQ на дашборде и алерт «В DLQ появились сообщения».

Повторы до delivery-limit показывает другая поломка — временная: остановленный
Postgres (docker compose stop postgres), тогда задача из API возвращается в очередь
и уходит в DLQ после трёх попыток.

    uv run python -m demo.poison
    uv run python -m demo.poison --count 3
"""

import argparse
import asyncio
import json
import uuid

from faststream.rabbit import RabbitBroker

from mlops_phishing.config import get_settings
from mlops_phishing.messaging import PUBLISH_CHANNEL, TASKS_QUEUE

# url — число, task_id и прочих полей нет: TaskMessage не провалидируется
POISON_BODY = {"url": 42}


def poison_id() -> str:
    """Метка сообщения: по ней его ищут в логах (task_id) и трейсах (task.id)."""
    return f"poison-{uuid.uuid4().hex[:8]}"


def poison_body() -> bytes:
    return json.dumps(POISON_BODY).encode()


def where_to_look(task_id: str) -> list[str]:
    """Подсказка: запросы, по которым видна история сообщения."""
    return [
        f'Loki:  {{service="worker"}} | task_id="{task_id}"',
        f'Tempo: {{span.task.id="{task_id}"}}',
        "Дашборд «Конвейер»: «DLQ сейчас», «Повторные доставки и перевод в DLQ»",
        "Алерт «В DLQ появились сообщения»: Firing примерно через 30–40 с",
    ]


async def publish_poison(count: int) -> list[str]:
    ids = [poison_id() for _ in range(count)]
    settings = get_settings()
    async with RabbitBroker(settings.rabbitmq_url, default_channel=PUBLISH_CHANNEL) as broker:
        for task_id in ids:
            # correlation_id = task_id, как у настоящих задач: воркер кладёт его в контекст логов
            await broker.publish(
                poison_body(),
                queue=TASKS_QUEUE,
                message_id=task_id,
                correlation_id=task_id,
                content_type="application/json",
            )
    return ids


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Битое сообщение в очередь tasks.")
    parser.add_argument("--count", type=int, default=1, help="сколько сообщений (1)")
    args = parser.parse_args(argv)
    for task_id in asyncio.run(publish_poison(args.count)):
        print(f"Отправлено битое сообщение {task_id}")
        for line in where_to_look(task_id):
            print(f"  {line}")


if __name__ == "__main__":
    main()
