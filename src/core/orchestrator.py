"""Orchestrator: creates engines, wires shared queues, and starts everything."""

from multiprocessing import Process, Queue
from typing import Optional

from src.engines.full_state_autoaim import FullStateAutoAimEngine
from src.engines.plot_engine import start_plot_engine
from src.toolbox.globals import config
from src.toolbox.logger import start_log_listener, stop_log_listener


def launch_system(engine_classes: list, engine_kwargs: Optional[dict] = None) -> list:
    """Spawn one shared driver process per declared driver, then start engines.

    Reads each engine class's ``drivers`` declaration, provisions + spawns one
    shared driver process per unique name (always shared — no in-process path),
    and starts every engine wired to those drivers via a ``driver_registry``.
    Returns all started processes (drivers + engines) for join/stop. This is the
    early config-driven orchestrator/watchdog described in CLAUDE.md.

    Engine classes used here must accept a ``driver_registry`` keyword.
    ``engine_kwargs`` optionally maps an engine class to extra constructor
    kwargs (e.g. a plot ``Queue``).
    """
    # One log queue + listener for all processes: children enqueue records,
    # this (parent) process owns the console/file handlers. Idempotent.
    log_queue, _ = start_log_listener()
    engine_kwargs = engine_kwargs or {}

    # One shared driver per unique name across all engines.
    needed: dict = {}
    for engine_cls in engine_classes:
        for name, driver_type in getattr(engine_cls, "drivers", {}).items():
            needed[name] = driver_type

    registry: dict = {}
    processes: list = []
    for name, driver_type in needed.items():
        conn = driver_type.provision(name)  # parent-side transport allocation
        registry[name] = conn
        driver = driver_type.from_conn(conn)
        driver.start()
        processes.append(driver)

    for engine_cls in engine_classes:
        engine = engine_cls(driver_registry=registry, **engine_kwargs.get(engine_cls, {}))
        engine._log_queue = log_queue  # inherited by the child at fork (see Engine.run)
        engine.start()
        processes.append(engine)

    return processes


def start_engines() -> None:
    """Start the production engines via the driver-aware factory.

    The engine declares `drivers = {"frames": CameraDriver, "mcu": McuDriver}`, so
    it must go through launch_system (which spawns the shared drivers and wires
    the registry).

    When ``config.plot.enable`` is true, also spawns the live matplotlib plot
    (``src.engines.plot_engine.start_plot_engine``) in its own process and wires
    a ``Queue`` into the auto-aim engine so it can stream the locked target's
    turret-frame x position to it.
    """
    engine_kwargs: dict = {}
    processes: list = []
    if config.plot.enable:
        plot_queue: Queue = Queue()
        plot_process = Process(
            target=start_plot_engine,
            args=(plot_queue, "Target panel turret-frame x (cm)"),
            daemon=True,
        )
        plot_process.start()
        processes.append(plot_process)
        engine_kwargs[FullStateAutoAimEngine] = {"queue": plot_queue}

    processes.extend(launch_system([FullStateAutoAimEngine], engine_kwargs=engine_kwargs))
    try:
        for p in processes:
            p.join()
    except KeyboardInterrupt:
        for p in processes:
            p.terminate()
            p.join()
    finally:
        stop_log_listener()  # flush any queued records before exit
