"""Тесты настройки логирования."""

import logging

from mlops_phishing.logging_config import (
    NO_REQUEST_ID,
    REQUEST_ID,
    RequestIdFilter,
    get_request_id,
    setup_logging,
)


def make_record() -> logging.LogRecord:
    """Собрать минимальную запись лога для проверки фильтра."""
    return logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="msg",
        args=(),
        exc_info=None,
    )


def test_filter_injects_request_id_from_context():
    """Идентификатор запроса из контекста попадает в запись лога."""
    token = REQUEST_ID.set("abcd1234")
    record = make_record()
    try:
        assert RequestIdFilter().filter(record) is True
    finally:
        REQUEST_ID.reset(token)

    assert get_request_id(record) == "abcd1234"


def test_filter_uses_placeholder_outside_request():
    """Вне обработки запроса в логе стоит прочерк, а не пустота."""
    record = make_record()

    assert RequestIdFilter().filter(record) is True
    assert get_request_id(record) == NO_REQUEST_ID


def test_setup_logging_installs_single_handler_and_level():
    """Настройка ставит один обработчик и заданный уровень."""
    setup_logging("DEBUG")
    root = logging.getLogger()

    assert len(root.handlers) == 1
    assert root.level == logging.DEBUG


def test_setup_logging_takes_over_uvicorn_loggers():
    """Логгеры uvicorn подчиняются общему формату, иначе в потоке два стиля."""
    setup_logging("INFO")
    root_handler = logging.getLogger().handlers[0]

    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        uvicorn_logger = logging.getLogger(name)
        assert uvicorn_logger.handlers == [root_handler]
        assert uvicorn_logger.propagate is False
