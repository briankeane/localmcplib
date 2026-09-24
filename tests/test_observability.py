from __future__ import annotations

import logging
import sys
import warnings
from collections.abc import Iterator
from io import StringIO
from pathlib import Path
from typing import Any

import pytest

import localmcp.observability.logging as logging_module
from localmcp.observability.logging import configure_logging, get_logger, resolve_log_level


@pytest.fixture(autouse=True)
def restore_logging_state() -> Iterator[None]:
    """Keep tests that exercise process-wide logging configuration isolated."""
    root = logging.getLogger()
    root_state = (list(root.handlers), root.level, root.propagate, root.disabled)
    logger_states = {
        name: (list(logger.handlers), logger.level, logger.propagate, logger.disabled)
        for name, logger in logging.root.manager.loggerDict.items()
        if isinstance(logger, logging.Logger)
    }
    last_resort = logging.lastResort
    raise_exceptions = logging.raiseExceptions
    showwarning = warnings.showwarning
    yield

    current_loggers = (item for item in logging.root.manager.loggerDict.values() if isinstance(item, logging.Logger))
    for logger in [root, *current_loggers]:
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
            if handler not in root_state[0] and all(handler not in state[0] for state in logger_states.values()):
                handler.close()
    root.handlers[:] = root_state[0]
    root.setLevel(root_state[1])
    root.propagate = root_state[2]
    root.disabled = root_state[3]
    for name, state in logger_states.items():
        logger = logging.getLogger(name)
        logger.handlers[:] = state[0]
        logger.setLevel(state[1])
        logger.propagate = state[2]
        logger.disabled = state[3]
    logging.lastResort = last_resort
    logging.raiseExceptions = raise_exceptions
    warnings.showwarning = showwarning


def test_log_level_precedence() -> None:
    assert resolve_log_level({"APP_LEVEL": "debug"}, env_var="APP_LEVEL", file_level="ERROR") == logging.DEBUG
    assert resolve_log_level({}, env_var="APP_LEVEL", file_level="ERROR") == logging.ERROR
    assert resolve_log_level({"APP_LEVEL": "invalid"}, env_var="APP_LEVEL", file_level="DEBUG") == logging.WARNING
    assert (
        resolve_log_level({"APP_LEVEL": "  "}, env_var="APP_LEVEL", file_level="  ", default=logging.INFO)
        == logging.INFO
    )


def test_configured_logger_writes_json_to_file(tmp_path: Path) -> None:
    path = tmp_path / "app.log"
    configure_logging(log_file=path, level=logging.INFO, managed_loggers=())

    get_logger("test").info("event.name", answer=42)

    rendered = path.read_text()
    assert '"event": "event.name"' in rendered
    assert '"answer": 42' in rendered


def test_configuration_failure_uses_null_handler_and_cleans_managed_logger(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class RefusesToClose(logging.Handler):
        def emit(self, _record: logging.LogRecord) -> None:
            pass

        def close(self) -> None:
            raise RuntimeError("close failed")

    def fail_to_open(*_args: Any, **_kwargs: Any) -> None:
        raise OSError("read-only filesystem")

    managed = logging.getLogger("test.managed")
    managed.addHandler(RefusesToClose())
    managed.setLevel(logging.ERROR)
    managed.propagate = False
    monkeypatch.setattr(logging_module, "TimedRotatingFileHandler", fail_to_open)

    configure_logging(
        log_file=tmp_path / "unavailable" / "app.log",
        level=logging.DEBUG,
        managed_loggers=(managed.name,),
    )

    root = logging.getLogger()
    assert root.level == logging.DEBUG
    assert len(root.handlers) == 1
    assert isinstance(root.handlers[0], logging.NullHandler)
    assert root.handlers[0].get_name() == "localmcp-file"
    assert managed.handlers == []
    assert managed.level == logging.NOTSET
    assert managed.propagate is True
    assert isinstance(logging.lastResort, logging.NullHandler)
    assert logging.raiseExceptions is False


def test_lazy_logger_removes_stdio_handlers_and_installs_safe_route(monkeypatch: pytest.MonkeyPatch) -> None:
    class RefusesToClose(logging.StreamHandler[Any]):
        def close(self) -> None:
            raise RuntimeError("close failed")

    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
    root.addHandler(RefusesToClose(sys.stderr))
    child = logging.getLogger("test.safe.child")
    child.addHandler(logging.StreamHandler(sys.stdout))

    configure_calls = 0
    original_configure = logging_module._configure_structlog

    def record_configure() -> None:
        nonlocal configure_calls
        configure_calls += 1
        original_configure()

    monkeypatch.setattr(logging_module.structlog, "is_configured", lambda: False)
    monkeypatch.setattr(logging_module, "_configure_structlog", record_configure)

    get_logger(child.name).warning("safe.event")

    assert configure_calls == 1
    assert child.handlers == []
    assert len(root.handlers) == 1
    assert isinstance(root.handlers[0], logging.NullHandler)
    assert root.handlers[0].get_name() == "localmcp-file"


def test_file_handlers_and_non_stdio_streams_are_recognized_as_safe(tmp_path: Path) -> None:
    file_handler = logging.FileHandler(tmp_path / "safe.log")
    non_stdio = logging.StreamHandler[Any](StringIO())

    assert logging_module._is_stdio_handler(file_handler) is False
    assert logging_module._is_stdio_handler(non_stdio) is False

    file_handler.close()
    non_stdio.close()
