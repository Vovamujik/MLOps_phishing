"""Структурные логи: одна JSON-строка на запись в stdout, единая для api, worker и source.

Пример: {"timestamp": "2026-11-11T09:30:01.123Z", "level": "info", "service": "worker",
"env": "development", "logger": "mlops_phishing.worker", "message": "Задача обработана",
"task_id": "…", "attempt": 1}

Почему structlog (через stdlib), а не другое:
- python-json-logger — только форматтер. Контекст (task_id, request_id), trace_id и
  маскирование паролей пришлось бы дописывать фильтрами вокруг него;
- свой logging.Formatter — json.dumps плюс разбор extra, exc_info и contextvars: те же
  полсотни строк, которые structlog уже написал и покрыл тестами;
- structlog.stdlib.ProcessorFormatter прогоняет через одну цепочку процессоров и наши
  записи, и чужие (uvicorn, faststream, aio_pika, asyncpg, mlflow, opentelemetry). Контекст
  задачи — встроенный merge_contextvars, процессор — обычная функция. Код приложения пишет
  стандартным logging, поэтому библиотеки ничего не знают о формате.

Контекст живёт в contextvars: request_id ставит middleware API, task_id — middleware
воркера на время сообщения, trace_id/span_id — из текущего спана OpenTelemetry.
Поля записи попадают в JSON из contextvars и из extra.
Уровень задаёт LOG_LEVEL; хранилище логов приложение не знает — только stdout.
"""

import logging
import os
import re
import sys
from collections.abc import MutableMapping
from datetime import UTC, datetime
from typing import Any

import structlog
import uvicorn.config
from opentelemetry import trace
from structlog.typing import EventDict, Processor, WrappedLogger

# Библиотеки, которые ставят себе свои обработчики: их записи отправляем в корневой.
# mlflow вешает текстовый обработчик при импорте (ml импортируется раньше настройки)
LIBRARY_LOGGERS = ("uvicorn", "uvicorn.error", "faststream", "faststream.access", "mlflow")
# Access-лог uvicorn дублирует middleware API, где есть request_id
SILENCED_LOGGERS = ("uvicorn.access",)

# Поля, которые идут первыми: строку удобно читать и глазами
FIRST_FIELDS = ("timestamp", "level", "service", "env", "logger", "message")
# Служебные поля библиотек, которым не место в JSON (uvicorn кладёт копию с ANSI-цветами)
DROPPED_FIELDS = ("color_message",)

# Логин:пароль в URL (amqp://, postgresql://, http://): пароль заменяется звёздочками
CREDENTIALS_IN_URL = re.compile(r"(?P<prefix>[a-z][a-z0-9+.-]*://[^:/@\s]+:)[^@\s]+@", re.I)
MASK = "***"


def mask_credentials(text: str) -> str:
    """Скрыть пароль в строках подключения, где бы он ни встретился в тексте."""
    return CREDENTIALS_IN_URL.sub(rf"\g<prefix>{MASK}@", text)


def add_timestamp(_: WrappedLogger, __: str, event_dict: EventDict) -> EventDict:
    """ISO-8601 в UTC с миллисекундами: так время сравнимо между контейнерами."""
    now = datetime.now(UTC)
    event_dict["timestamp"] = now.strftime("%Y-%m-%dT%H:%M:%S.") + f"{now.microsecond // 1000:03d}Z"
    return event_dict


def add_trace_ids(_: WrappedLogger, __: str, event_dict: EventDict) -> EventDict:
    """trace_id и span_id текущего спана: из лога одним кликом — в трейс в Tempo."""
    context = trace.get_current_span().get_span_context()
    if context.is_valid:
        event_dict["trace_id"] = trace.format_trace_id(context.trace_id)
        event_dict["span_id"] = trace.format_span_id(context.span_id)
    return event_dict


def static_fields(service: str, env: str) -> Processor:
    """Поля процесса: по ним Loki режет потоки (лейблы service, env)."""

    def add(_: WrappedLogger, __: str, event_dict: EventDict) -> EventDict:
        event_dict["service"] = service
        event_dict["env"] = env
        return event_dict

    return add


def scrub(_: WrappedLogger, __: str, event_dict: EventDict) -> EventDict:
    """Убрать служебные поля и пароли из всех строковых значений, включая стектрейс."""
    for name in DROPPED_FIELDS:
        event_dict.pop(name, None)
    for key, value in event_dict.items():
        if isinstance(value, str):
            event_dict[key] = mask_credentials(value)
    return event_dict


def order_fields(_: WrappedLogger, __: str, event_dict: EventDict) -> MutableMapping[str, Any]:
    """Основные поля в начале строки, остальные — в порядке появления."""
    head = {name: event_dict.pop(name) for name in FIRST_FIELDS if name in event_dict}
    return head | event_dict


def build_formatter(service: str, env: str) -> structlog.stdlib.ProcessorFormatter:
    """Форматтер для корневого обработчика: запись logging → JSON."""
    pre_chain: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.ExtraAdder(),
        add_trace_ids,
        add_timestamp,
        static_fields(service, env),
    ]
    return structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=pre_chain,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            structlog.processors.format_exc_info,
            structlog.processors.EventRenamer("message"),
            scrub,
            order_fields,
            structlog.processors.JSONRenderer(ensure_ascii=False, default=str),
        ],
    )


def route_uvicorn_logs() -> None:
    """Не дать uvicorn переписать настройку логов после нас.

    `faststream run` сначала импортирует приложение, а потом создаёт uvicorn.Config,
    который применяет свой dictConfig и вешает на логгеры uvicorn текстовые обработчики.
    Config берёт словарь LOGGING_CONFIG по ссылке, поэтому подменяем его содержимое:
    логгеры uvicorn без своих обработчиков пишут через корневой, то есть в JSON.
    """
    uvicorn.config.LOGGING_CONFIG.clear()
    uvicorn.config.LOGGING_CONFIG.update(
        {
            "version": 1,
            "disable_existing_loggers": False,
            # Ключи нужны uvicorn: при --use-colors он пишет в них флаг
            "formatters": {"default": {}, "access": {}},
            "loggers": {
                "uvicorn": {"handlers": [], "propagate": True},
                "uvicorn.error": {"handlers": [], "propagate": True},
                "uvicorn.access": {"handlers": [], "propagate": False},
            },
        }
    )


def setup_logging(level: str, service: str, env: str) -> None:
    """Один JSON-обработчик на корневом логгере; библиотеки пишут через него.

    Имя сервиса — из OTEL_SERVICE_NAME, если задано: тогда поле `service` в логах и
    service.name в трейсах и метриках совпадают.
    """
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(build_formatter(os.environ.get("OTEL_SERVICE_NAME") or service, env))

    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level.upper())

    for name in LIBRARY_LOGGERS:
        library_logger = logging.getLogger(name)
        library_logger.handlers = []
        library_logger.propagate = True
    for name in SILENCED_LOGGERS:
        logging.getLogger(name).disabled = True
    route_uvicorn_logs()
