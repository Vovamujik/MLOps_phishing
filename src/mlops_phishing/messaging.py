"""Очереди конвейера задач: одно описание для API, источника, воркера и тестов.

Аргументы очереди в RabbitMQ неизменяемы: передекларация с другими значениями
даёт PRECONDITION_FAILED. Поэтому здесь только тип очереди, а лимит длины,
политика переполнения и DLQ задаются policy (см. rabbit_setup).
"""

import uuid
from datetime import datetime
from typing import Literal

from faststream.rabbit import QueueType, RabbitBroker, RabbitQueue
from faststream.rabbit.schemas import Channel
from pydantic import BaseModel, Field

TASKS_QUEUE_NAME = "tasks"
DEAD_LETTER_EXCHANGE = "tasks.dlx"
DEAD_LETTER_QUEUE_NAME = "tasks.dlq"

# Quorum: только у неё есть счётчик доставок (delivery-limit) для перехода в DLQ
TASKS_QUEUE = RabbitQueue(TASKS_QUEUE_NAME, durable=True, queue_type=QueueType.QUORUM)

# Канал издателей (источник, API). publisher_confirms: на каждую публикацию брокер отвечает
# ack или nack, nack при полной очереди — DeliveryError. on_return_raises: если очереди нет,
# брокер возвращает сообщение, и без флага FastStream молча считает его отправленным
PUBLISH_CHANNEL = Channel(publisher_confirms=True, on_return_raises=True)


class TaskMessage(BaseModel):
    """Задача в очереди: одна URL-строка на скоринг."""

    task_id: uuid.UUID
    url: str = Field(min_length=1, max_length=2048)
    source: Literal["api", "batch"]
    # Момент постановки в очередь: от него считается время выполнения задачи
    enqueued_at: datetime


async def publish_task(broker: RabbitBroker, task: TaskMessage) -> None:
    """Опубликовать задачу и дождаться ответа брокера.

    Брокер должен быть открыт с PUBLISH_CHANNEL. Очередь полна — DeliveryError (nack),
    очереди нет — PublishError (mandatory по умолчанию). persist для наглядности:
    quorum-очередь и так пишет всё на диск, а сохранность гарантирует confirm — ack
    приходит после записи на большинство узлов.
    """
    await broker.publish(task, queue=TASKS_QUEUE, message_id=str(task.task_id), persist=True)
