"""Настройка логирования.

Формат строки:
`2026-09-14T17:55:01.123 | INFO     | a1b2c3d4 | mlops_phishing.app | ...`

Третье поле — `request_id`. Он живёт в ContextVar, который middleware ставит
на время обработки запроса, поэтому все записи одного запроса связаны общим
идентификатором. Без этого в конкурентном асинхронном приложении логи разных
запросов перемешаны и отладка превращается в гадание.
"""

import contextvars
import logging
import sys

REQUEST_ID: contextvars.ContextVar[str | None] = contextvars.ContextVar("request_id", default=None)

LOG_FORMAT = "%(asctime)s.%(msecs)03d | %(levelname)-8s | %(request_id)-8s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%dT%H:%M:%S"

NO_REQUEST_ID = "-"


class RequestIdFilter(logging.Filter):
    """Подмешивает request_id из контекста в каждую запись лога."""

    def filter(self, record: logging.LogRecord) -> bool:
        """Добавить атрибут request_id и пропустить запись дальше."""
        record.request_id = REQUEST_ID.get() or NO_REQUEST_ID
        return True


def get_request_id(record: logging.LogRecord) -> str:
    """Прочитать идентификатор запроса, подставленный фильтром.

    Атрибут появляется у записи динамически, поэтому читается через getattr
    с дефолтом: так контракт фильтра описан в одном месте, а не размазан
    по местам чтения.
    """
    return getattr(record, "request_id", NO_REQUEST_ID)


def setup_logging(level: str) -> None:
    """Настроить корневой логгер и логгеры uvicorn в едином формате."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))
    handler.addFilter(RequestIdFilter())

    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level.upper())

    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        uvicorn_logger = logging.getLogger(name)
        uvicorn_logger.handlers = [handler]
        uvicorn_logger.propagate = False
