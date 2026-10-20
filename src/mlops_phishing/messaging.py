"""Очереди конвейера задач: одно описание для API, источника, воркера и тестов.

Аргументы очереди в RabbitMQ неизменяемы: передекларация с другими значениями
даёт PRECONDITION_FAILED. Поэтому здесь только тип очереди, а лимит длины,
политика переполнения и DLQ задаются policy (см. rabbit_setup).
"""

import uuid
from datetime import datetime
from typing import Literal

from faststream.rabbit import QueueType, RabbitQueue
from pydantic import BaseModel, Field

TASKS_QUEUE_NAME = "tasks"
DEAD_LETTER_EXCHANGE = "tasks.dlx"
DEAD_LETTER_QUEUE_NAME = "tasks.dlq"

# Quorum: только у неё есть счётчик доставок (delivery-limit) для перехода в DLQ
TASKS_QUEUE = RabbitQueue(TASKS_QUEUE_NAME, durable=True, queue_type=QueueType.QUORUM)


class TaskMessage(BaseModel):
    """Задача в очереди: одна URL-строка на скоринг."""

    task_id: uuid.UUID
    url: str = Field(min_length=1, max_length=2048)
    source: Literal["api", "batch"]
    # Момент постановки в очередь: от него считается время выполнения задачи
    enqueued_at: datetime
