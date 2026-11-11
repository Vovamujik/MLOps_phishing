"""Задачи: приём URL в очередь и результат скоринга.

POST /tasks — граница системы. Внешнего клиента замедлить нельзя, поэтому при
полной очереди API не ждёт места, а сразу отвечает 503 с Retry-After. Это сброс
нагрузки (load shedding), а не противодавление: противодавление замедляет того,
кто внутри контура (источник CSV ждёт и повторяет ту же задачу), а здесь лишняя
работа отбрасывается на входе, и повторить или нет — решает клиент.

Порядок записи: сначала публикация, потом строка PENDING в базе.
- брокер отказал — в базу ничего не пишется, после 503 PENDING не остаётся;
- брокер принял, а база недоступна — задача не потеряна: воркер создаст строку
  сам, когда запишет результат (upsert), поэтому ответ всё равно 202 (до этого
  GET отвечает 404);
- брокер не ответил за таймаут — исход неизвестен: публикация могла уйти, тогда
  воркер запишет DONE сам (at-least-once). PENDING не остаётся и здесь, но повтор
  клиента — уже новая задача;
- обратный порядок оставлял бы PENDING-сирот при отказе брокера и при падении
  API между двумя записями.
"""

import logging
import math
import uuid
from datetime import UTC, datetime

import structlog
from aio_pika.exceptions import DeliveryError, PublishError
from fastapi import APIRouter, HTTPException, Request, Response, status
from opentelemetry import trace
from opentelemetry.trace import StatusCode

from mlops_phishing import tasks_repo
from mlops_phishing.config import Settings
from mlops_phishing.db import DEPENDENCY_ERRORS
from mlops_phishing.messaging import TaskMessage
from mlops_phishing.publisher import BROKER_ERRORS, TaskPublisher
from mlops_phishing.schemas import TaskAccepted, TaskCreate, TaskView
from mlops_phishing.tasks_repo import TaskStatus

log = logging.getLogger(__name__)

router = APIRouter(tags=["tasks"])

# Quorum-очередь после отказа снова принимает, когда готовых задач 80% лимита
REOPEN_SHARE = 0.8


def retry_after_seconds(settings: Settings) -> int:
    """Через сколько очередь снова начнёт принимать задачи.

    После отказа воркеры должны разобрать 20% лимита. Воркер считает одновременно
    min(пул, prefetch) задач, то есть разбирает min(пул, prefetch) / inference_delay
    задач в секунду: при лимите 100, 4 потоках, prefetch 4 и 0.2 с это 20 задач при
    20 задачах/с, то есть 1 с. Это оценка при скорости одного воркера, а не гарантия:
    воркеров больше — место освобождается быстрее, но освободившиеся места разбирает
    и источник. В заголовке целые секунды, минимум 1.
    """
    backlog = (1 - REOPEN_SHARE) * settings.tasks_queue_max_length
    # prefetch 0 — без ограничения: одновременно считается столько, сколько потоков
    concurrency = min(
        settings.worker_pool_size, settings.worker_prefetch or settings.worker_pool_size
    )
    seconds = backlog * settings.inference_delay_seconds / concurrency
    # round: двоичная погрешность дробей не должна сдвигать целые секунды вверх
    return max(1, math.ceil(round(seconds, 3)))


def mark_failed(span: trace.Span, exc: BaseException, reason: str) -> None:
    """503 в трейсе: статус ERROR и само исключение брокера на спане запроса."""
    span.record_exception(exc)
    span.set_status(StatusCode.ERROR, reason)


@router.post(
    "/tasks",
    status_code=status.HTTP_202_ACCEPTED,
    summary="Поставить URL на проверку",
    description="Задача уходит в очередь, результат — по ссылке из заголовка Location.",
    responses={
        status.HTTP_503_SERVICE_UNAVAILABLE: {
            "description": "Очередь заполнена (с заголовком Retry-After) или брокер недоступен"
        }
    },
)
async def create_task(payload: TaskCreate, request: Request, response: Response) -> TaskAccepted:
    settings: Settings = request.app.state.settings
    publisher: TaskPublisher = request.app.state.publisher
    task = TaskMessage(
        task_id=uuid.uuid4(), url=payload.url, source="api", enqueued_at=datetime.now(UTC)
    )
    # Все записи запроса дальше — с task_id; access-лог middleware берёт его из state
    structlog.contextvars.bind_contextvars(task_id=str(task.task_id))
    request.state.task_id = task.task_id
    # По task.id трейс ищется в Tempo: {span.task.id = "..."}
    span = trace.get_current_span()
    span.set_attribute("task.id", str(task.task_id))

    try:
        await publisher.publish(task)
    except PublishError as exc:
        mark_failed(span, exc, "очереди задач нет")
        log.error("Очереди задач нет: 503", exc_info=exc)
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE, "Очередь задач не настроена"
        ) from exc
    except DeliveryError as exc:
        # Брокер ответил nack: очередь полна. Ждать места нельзя — отказ сразу.
        # WARNING, а не ERROR: это штатный сброс нагрузки, но его должно быть видно
        seconds = retry_after_seconds(settings)
        mark_failed(span, exc, "очередь заполнена")
        log.warning(
            "Очередь заполнена: 503, Retry-After %d с", seconds, extra={"retry_after": seconds}
        )
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            f"Очередь задач заполнена, повторите через {seconds} с",
            headers={"Retry-After": str(seconds)},
        ) from exc
    except BROKER_ERRORS as exc:
        # После таймаута исход неизвестен: задача могла уйти (см. docstring модуля)
        mark_failed(span, exc, "брокер недоступен")
        log.error("Брокер недоступен: 503", exc_info=exc)
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Брокер задач недоступен") from exc

    try:
        await tasks_repo.insert_pending(
            request.app.state.db_pool, task.task_id, task.url, task.enqueued_at
        )
    except DEPENDENCY_ERRORS as exc:
        log.warning("Задача в очереди, строка PENDING не записана: %s", type(exc).__name__)

    status_url = str(request.url_for("read_task", task_id=str(task.task_id)))
    response.headers["Location"] = status_url
    return TaskAccepted(task_id=task.task_id, status=TaskStatus.PENDING, status_url=status_url)


@router.get(
    "/tasks/{task_id}",
    summary="Статус и результат задачи",
    responses={
        status.HTTP_404_NOT_FOUND: {"description": "Задачи нет или она ещё не записана"},
        status.HTTP_503_SERVICE_UNAVAILABLE: {"description": "База данных недоступна"},
    },
)
async def read_task(task_id: uuid.UUID, request: Request) -> TaskView:
    try:
        record = await tasks_repo.get_task(request.app.state.db_pool, task_id)
    except DEPENDENCY_ERRORS as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "База данных недоступна") from exc
    if record is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Задача не найдена")
    return TaskView.model_validate(record)
