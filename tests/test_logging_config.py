"""Тесты логов: форма JSON, контекст, маскирование паролей, логгеры библиотек."""

import logging
import re

import structlog
import uvicorn

from mlops_phishing.logging_config import mask_credentials, setup_logging

TIMESTAMP = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$")
log = logging.getLogger("mlops_phishing.test")


def test_record_is_one_json_line_with_contract_fields(json_log):
    log.info("Модель загружена: %s", "v3", extra={"model_version": "3"})

    [record] = json_log.find("Модель загружена")
    assert list(record)[:6] == ["timestamp", "level", "service", "env", "logger", "message"]
    assert TIMESTAMP.match(str(record["timestamp"]))
    assert record["level"] == "info"
    assert record["logger"] == "mlops_phishing.test"
    assert record["message"] == "Модель загружена: v3"
    assert (record["service"], record["env"]) == ("test-service", "test")
    assert record["model_version"] == "3"


def test_context_fields_come_from_contextvars(json_log):
    """task_id не передаётся в вызов лога — его кладёт контекст."""
    with structlog.contextvars.bound_contextvars(task_id="t-1", request_id="r-1"):
        log.warning("внутри задачи")
    log.warning("снаружи")

    [inside] = json_log.find("внутри задачи")
    [outside] = json_log.find("снаружи")
    assert (inside["task_id"], inside["request_id"], inside["level"]) == ("t-1", "r-1", "warning")
    assert "task_id" not in outside


def test_exception_is_a_string_with_traceback(json_log):
    try:
        raise ValueError("boom")
    except ValueError:
        log.exception("ошибка обработки")

    [record] = json_log.find("ошибка обработки")
    assert record["level"] == "error"
    assert "Traceback" in str(record["exception"])
    assert "ValueError: boom" in str(record["exception"])


def test_passwords_in_urls_never_reach_the_log(json_log):
    """Пароль из DSN или URL брокера скрыт и в сообщении, и в стектрейсе."""
    url = "amqp://app:s3cret@rabbitmq:5672/"
    try:
        raise ConnectionError(f"cannot connect to {url}")
    except ConnectionError:
        log.exception("Брокер %s недоступен", url, extra={"dsn": "postgresql://u:pw1@db/x"})

    assert "s3cret" not in json_log.text
    assert "pw1" not in json_log.text
    assert "amqp://app:***@rabbitmq:5672/" in json_log.text


def test_mask_keeps_urls_without_credentials():
    assert mask_credentials("http://mlflow:5000/api") == "http://mlflow:5000/api"
    assert mask_credentials("postgresql://u:p@db:5432/x") == "postgresql://u:***@db:5432/x"


def test_library_service_fields_are_dropped(json_log):
    """uvicorn кладёт в extra копию сообщения с ANSI-цветами: в JSON ей не место."""
    logging.getLogger("uvicorn.error").info("Started", extra={"color_message": "\x1b[1mStarted"})

    [record] = json_log.find("Started")
    assert "color_message" not in record


def test_setup_logging_installs_single_json_handler_and_level(capsys):
    setup_logging("DEBUG", "worker", "development")
    root = logging.getLogger()

    assert len(root.handlers) == 1
    assert root.level == logging.DEBUG
    logging.getLogger("faststream").info("FastStream app started")
    assert '"service": "worker"' in capsys.readouterr().out


def test_service_name_follows_otel_service_name(monkeypatch, capsys):
    """Поле service совпадает с service.name трейсов и метрик."""
    monkeypatch.setenv("OTEL_SERVICE_NAME", "worker-2")
    setup_logging("INFO", "worker", "development")

    log.info("hello")
    assert '"service": "worker-2"' in capsys.readouterr().out


def test_library_loggers_write_through_root(monkeypatch):
    """У uvicorn и FastStream свои текстовые обработчики: их снимаем, пишут через корневой."""
    faststream_logger = logging.getLogger("faststream")
    monkeypatch.setattr(faststream_logger, "handlers", [logging.StreamHandler()])
    monkeypatch.setattr(faststream_logger, "propagate", False)

    setup_logging("INFO", "api", "development")

    for name in ("uvicorn", "uvicorn.error", "faststream"):
        assert logging.getLogger(name).handlers == []
        assert logging.getLogger(name).propagate is True


def test_uvicorn_config_created_later_keeps_json(monkeypatch):
    """`faststream run` создаёт uvicorn.Config после импорта приложения — формат не ломается."""
    setup_logging("INFO", "worker", "development")
    monkeypatch.setattr(logging.getLogger("uvicorn.error"), "handlers", [logging.StreamHandler()])

    uvicorn.Config(app=lambda: None, log_level="info")

    assert logging.getLogger("uvicorn.error").handlers == []
    assert logging.getLogger("uvicorn.error").propagate is True
    # Access-лог uvicorn не пишется: его заменяет middleware API
    assert logging.getLogger("uvicorn.access").propagate is False
    assert logging.getLogger("uvicorn.access").handlers == []


def test_setup_logging_disables_uvicorn_access_log(monkeypatch):
    access_logger = logging.getLogger("uvicorn.access")
    monkeypatch.setattr(access_logger, "disabled", False)

    setup_logging("INFO", "api", "development")

    assert access_logger.disabled is True
