"""Multiprocess logging for AutoX (stdlib ``logging``, queue-based).

Engines and drivers are separate OS processes, so they cannot share file/console
handlers directly. This module implements the pattern from the Python logging
cookbook ("logging to a single file from multiple processes"):

- The **parent** (orchestrator) calls :func:`start_log_listener`. It owns the
  real handlers (console + optional rotating file) on a ``QueueListener``
  thread.
- Each **child** process calls :func:`configure_child_logging` with the shared
  queue; its records are enqueued and formatted/written by the parent. The
  engine base class does this automatically at the top of ``Engine.run()``.
- Code gets loggers via :func:`get_logger`; ``Module`` instances have
  ``self.log`` pre-made.

Config (``info.yaml`` ``log:`` block): ``level`` (root level), ``levels``
(per-logger overrides, e.g. ``{panel_tracking: DEBUG}`` — lets you debug one
module without drowning in another's output), ``to_file`` + the ``(path_to)
log_file`` entry, ``max_file_mb``/``backup_count``, and ``disable_all_logging``
as a kill switch (``logging.disable`` — disabled calls cost a near-free integer
check).

Hot-loop guidance: use lazy %-style args (``log.debug("x=%s", x)``, never
f-strings in per-frame paths) and wrap expensive formatting in
``log.isEnabledFor(logging.DEBUG)``.
"""

import logging
import logging.handlers
import multiprocessing as mp
from pathlib import Path
from typing import Dict, Optional, Tuple

_ROOT_NAME = "autox"
_FORMAT = "%(asctime)s %(levelname)-7s %(processName)s %(name)s: %(message)s"

#: Listener + queue owned by the parent process (set by start_log_listener).
_listener: Optional[logging.handlers.QueueListener] = None
_queue: Optional[mp.Queue] = None


def get_logger(name: str) -> logging.Logger:
    """Return the AutoX logger for ``name`` (namespaced under ``autox.``)."""
    logger = logging.getLogger(f"{_ROOT_NAME}.{name}")
    # ROS 2's launch.logging hijacks the global logger class (setLoggerClass)
    # and its loggers default to propagate=False. All our handlers hang off the
    # "autox" root, so force propagation or records silently vanish whenever
    # ROS is sourced (MCU=ROS, launch_testing under pytest, ...).
    logger.propagate = True
    return logger


def _log_config() -> dict:
    """Read the ``log:`` config block (empty dict if config is unavailable)."""
    from src.toolbox.globals import config  # local import: keep module import cheap

    return dict(config.log or {})


def _build_handlers() -> list:
    cfg = _log_config()
    console = logging.StreamHandler()
    handlers: list = [console]
    if cfg.get("to_file", False):
        from src.toolbox.globals import absolute_path_to

        path = Path(absolute_path_to.log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(
            logging.handlers.RotatingFileHandler(
                path,
                maxBytes=int(cfg.get("max_file_mb", 20)) * 1024 * 1024,
                backupCount=int(cfg.get("backup_count", 3)),
            )
        )
    formatter = logging.Formatter(_FORMAT)
    for handler in handlers:
        handler.setFormatter(formatter)
    return handlers


def start_log_listener() -> Tuple[mp.Queue, logging.handlers.QueueListener]:
    """Parent-side setup: start the listener thread that owns the real handlers.

    Also configures the parent's own loggers to go through the queue, so parent
    and children share one coherent output. Idempotent: a second call returns
    the existing listener's queue.
    """
    global _listener, _queue
    if _listener is not None and _queue is not None:
        return _queue, _listener

    _queue = mp.Queue()
    _listener = logging.handlers.QueueListener(_queue, *_build_handlers())
    _listener.start()
    configure_child_logging(_queue)
    return _queue, _listener


def stop_log_listener() -> None:
    """Flush and stop the parent-side listener (safe to call when not started)."""
    global _listener, _queue
    if _listener is not None:
        _listener.stop()
        _listener = None
        _queue = None


def configure_child_logging(
    queue: Optional[mp.Queue],
    level: Optional[str] = None,
    levels: Optional[Dict[str, str]] = None,
    disable_all: Optional[bool] = None,
) -> None:
    """Child-side setup: route all ``autox.*`` records into ``queue``.

    ``queue=None`` (unit tests, standalone scripts) attaches a plain console
    handler instead, so ``get_logger`` always works. The keyword overrides
    exist for tests; production reads the ``log:`` config block.
    """
    # Only read the global config for args the caller didn't supply -- tests
    # pass everything explicitly and must not trigger a config (CLI) parse.
    if level is None or levels is None or disable_all is None:
        cfg = _log_config()
        level = level if level is not None else cfg.get("level", "INFO")
        levels = levels if levels is not None else (cfg.get("levels") or {})
        disable_all = (
            disable_all if disable_all is not None else cfg.get("disable_all_logging", False)
        )
    assert levels is not None

    root = logging.getLogger(_ROOT_NAME)
    root.handlers.clear()
    if queue is None:
        handler: logging.Handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_FORMAT))
    else:
        handler = logging.handlers.QueueHandler(queue)
    root.addHandler(handler)
    root.propagate = False
    root.setLevel(level)

    for name, name_level in levels.items():
        get_logger(name).setLevel(name_level)

    if disable_all:
        logging.disable(logging.CRITICAL)
