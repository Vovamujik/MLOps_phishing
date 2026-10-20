"""Очереди конвейера задач: одно описание для API, источника, воркера и тестов.

Аргументы очереди в RabbitMQ неизменяемы: передекларация с другими значениями
даёт PRECONDITION_FAILED. Поэтому здесь только тип очереди, а лимит длины,
политика переполнения и DLQ задаются policy (см. rabbit_setup).
"""

from faststream.rabbit import QueueType, RabbitQueue

TASKS_QUEUE_NAME = "tasks"
DEAD_LETTER_EXCHANGE = "tasks.dlx"
DEAD_LETTER_QUEUE_NAME = "tasks.dlq"

# Quorum: только у неё есть счётчик доставок (delivery-limit) для перехода в DLQ
TASKS_QUEUE = RabbitQueue(TASKS_QUEUE_NAME, durable=True, queue_type=QueueType.QUORUM)
