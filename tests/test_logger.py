"""Unit tests for src.toolbox.logger (multiprocess queue logging)."""

import logging
import logging.handlers
import queue
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.toolbox.logger import configure_child_logging, get_logger  # noqa: E402


def _drain(listener):
    listener.stop()  # flushes everything still in the queue


def test_queue_round_trip_to_file(tmp_path):
    """A record logged in 'child' config reaches the listener's file handler."""
    log_file = tmp_path / "autox.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter("%(name)s %(levelname)s %(message)s"))

    q: queue.Queue = queue.Queue()
    listener = logging.handlers.QueueListener(q, file_handler)
    listener.start()
    try:
        configure_child_logging(q, level="INFO", levels={}, disable_all=False)
        get_logger("test_module").info("hello %s", "world")
    finally:
        _drain(listener)
        file_handler.close()

    contents = log_file.read_text()
    assert "autox.test_module INFO hello world" in contents


def test_per_logger_level_override(tmp_path):
    """Per-logger level overrides beat the root level in both directions."""
    log_file = tmp_path / "autox.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter("%(name)s %(message)s"))

    q: queue.Queue = queue.Queue()
    listener = logging.handlers.QueueListener(q, file_handler)
    listener.start()
    try:
        configure_child_logging(
            q, level="INFO", levels={"noisy": "WARNING", "quiet": "DEBUG"}, disable_all=False
        )
        get_logger("noisy").info("dropped")      # below its WARNING override
        get_logger("quiet").debug("kept")        # its DEBUG override beats root INFO
        get_logger("normal").debug("dropped")    # below root INFO
        get_logger("normal").info("kept too")
    finally:
        _drain(listener)
        file_handler.close()

    contents = log_file.read_text()
    assert "dropped" not in contents
    assert "autox.quiet kept" in contents
    assert "autox.normal kept too" in contents


def test_disable_all_kill_switch(tmp_path):
    """disable_all silences even CRITICAL records."""
    log_file = tmp_path / "autox.log"
    file_handler = logging.FileHandler(log_file)

    q: queue.Queue = queue.Queue()
    listener = logging.handlers.QueueListener(q, file_handler)
    listener.start()
    try:
        configure_child_logging(q, level="DEBUG", levels={}, disable_all=True)
        get_logger("anything").critical("should never appear")
    finally:
        logging.disable(logging.NOTSET)  # undo global state for other tests
        _drain(listener)
        file_handler.close()

    assert log_file.read_text() == ""


def test_no_queue_falls_back_to_console(capsys):
    """queue=None attaches a plain console handler (tests/scripts)."""
    configure_child_logging(None, level="INFO", levels={}, disable_all=False)
    get_logger("console_module").info("to stderr")

    captured = capsys.readouterr()
    assert "to stderr" in captured.err
