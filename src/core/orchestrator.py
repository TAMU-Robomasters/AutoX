"""Orchestrator: creates engines, wires shared queues, and starts everything."""

from src.toolbox.globals import config
from src.toolbox.logger import get_logger, start_log_listener, stop_log_listener

log = get_logger("orchestrator")


def launch_system(engine_classes: list) -> list:
    """Spawn one shared driver process per declared driver, then start engines.

    Reads each engine class's ``drivers`` declaration, provisions + spawns one
    shared driver process per unique name (always shared — no in-process path),
    and starts every engine wired to those drivers via a ``driver_registry``.
    Returns all started processes (drivers + engines) for join/stop. This is the
    early config-driven orchestrator/watchdog described in CLAUDE.md.

    Engine classes used here must accept a ``driver_registry`` keyword.
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
        engine = engine_cls(driver_registry=registry)
        engine._log_queue = log_queue  # inherited by the child at fork (see Engine.run)
        engine.start()
        processes.append(engine)

    return processes


def start_engines() -> None:
    """Start the production engines via the driver-aware factory.

    The engine declares `drivers = {"frames": CameraDriver, "mcu": McuDriver}`, so
    it must go through launch_system (which spawns the shared drivers and wires
    the registry).

    NOTE: the old angular-velocity plot (start_plot_engine + a shared Queue) is
    dropped here for now — launch_system doesn't thread the plot Queue through.
    Re-add later if needed (the engine already accepts an optional `queue`).

    The detection backend is chosen by ``config.vision_backend`` (``info.yaml``):
    ``cpp`` runs the C++ detector that owns the camera; ``python`` (default) runs
    the Python detector + shared CameraDriver.
    """
    from src.engines.full_state_autoaim import FullStateAutoAimEngine

    backend = str(getattr(config, "vision_backend", "python")).lower()
    engine_cls: type[FullStateAutoAimEngine]
    if backend == "cpp":
        from src.engines.full_state_autoaim_cpp import FullStateAutoAimEngineCpp

        engine_cls = FullStateAutoAimEngineCpp
    else:
        engine_cls = FullStateAutoAimEngine
    log.info("starting production engine: %s (vision_backend=%s)", engine_cls.__name__, backend)
    processes = launch_system([engine_cls])
    try:
        for p in processes:
            p.join()
    except KeyboardInterrupt:
        for p in processes:
            p.terminate()
            p.join()
    finally:
        stop_log_listener()  # flush any queued records before exit
