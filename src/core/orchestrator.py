"""Orchestrator: creates engines, wires shared queues, and starts everything."""

import inspect
from multiprocessing import Queue
from typing import Optional

from src.engines.full_state_autoaim import FullStateAutoAimEngine
from src.toolbox.globals import config
from src.toolbox.logger import get_logger, start_log_listener, stop_log_listener


def launch_system(engine_classes: list, plot_queue: Optional[Queue] = None) -> list:
    """Spawn one shared driver process per declared driver, then start engines.

    Reads each engine class's ``drivers`` declaration, provisions + spawns one
    shared driver process per unique name (always shared — no in-process path),
    and starts every engine wired to those drivers via a ``driver_registry``.
    Returns all started processes (drivers + engines) for join/stop. This is the
    early config-driven orchestrator/watchdog described in CLAUDE.md.

    Engine classes used here must accept a ``driver_registry`` keyword. If
    ``plot_queue`` is given it is also passed to any engine whose constructor
    accepts a ``queue`` keyword (the live debug plot — see ``start_engines``).
    """
    # One log queue + listener for all processes: children enqueue records,
    # this (parent) process owns the console/file handlers. Idempotent.
    log_queue, _ = start_log_listener()

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
        kwargs: dict = {"driver_registry": registry}
        if plot_queue is not None and "queue" in inspect.signature(engine_cls).parameters:
            kwargs["queue"] = plot_queue
        engine = engine_cls(**kwargs)
        engine._log_queue = log_queue  # inherited by the child at fork (see Engine.run)
        engine.start()
        processes.append(engine)

    return processes


def start_engines() -> None:
    """Start the production engines via the driver-aware factory.

    The engine declares `drivers = {"frames": CameraDriver, "mcu": McuDriver}`, so
    it must go through launch_system (which spawns the shared drivers and wires
    the registry).

    A live debug plot (``start_plot_engine``) runs in its own process: the engine
    pushes the turret-frame transformed panel x (cm) into a shared Queue and the
    plot shows a rolling window of it. Toggle via ``config.log.live_plot``.
    """
    import multiprocessing as mp
    import os

    log = get_logger("orchestrator")
    plot_proc = None
    plot_queue: Optional[Queue] = None
    if config.log.live_plot and not os.environ.get("DISPLAY"):
        log.warning("live_plot is on but $DISPLAY is unset; skipping the plot "
                    "(run on the Jetson desktop or with `ssh -X`).")
    elif config.log.live_plot:
        from src.engines.plot_engine import start_plot_engine

        plot_queue = mp.Queue(maxsize=4000)
        plot_proc = mp.Process(
            target=start_plot_engine,
            args=(plot_queue,),
            kwargs={"title": "Transformed panel x (cm)"},
            daemon=True,
        )
        plot_proc.start()

    processes = launch_system([FullStateAutoAimEngine], plot_queue=plot_queue)
    try:
        for p in processes:
            p.join()
    except KeyboardInterrupt:
        for p in processes:
            p.terminate()
            p.join()
    finally:
        if plot_proc is not None:
            plot_proc.terminate()
            plot_proc.join()
        stop_log_listener()  # flush any queued records before exit
