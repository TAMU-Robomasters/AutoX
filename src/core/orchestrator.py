"""Orchestrator: creates engines, wires shared queues, and starts everything."""

from src.engines.particle_filter_autoaim import ParticleFilterAutoAimEngine


def launch_system(engine_classes: list) -> list:
    """Spawn one shared driver process per declared driver, then start engines.

    Reads each engine class's ``drivers`` declaration, provisions + spawns one
    shared driver process per unique name (always shared — no in-process path),
    and starts every engine wired to those drivers via a ``driver_registry``.
    Returns all started processes (drivers + engines) for join/stop. This is the
    early config-driven orchestrator/watchdog described in CLAUDE.md.

    Engine classes used here must accept a ``driver_registry`` keyword.
    """
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
        engine.start()
        processes.append(engine)

    return processes


def start_engines() -> None:
    """Start the production engines via the driver-aware factory.

    PF now declares `drivers = {"frames": CameraDriver}`, so it must go through
    launch_system (which spawns the shared CameraDriver and wires the registry).

    NOTE: the old angular-velocity plot (start_plot_engine + a shared Queue) is
    dropped here for now — launch_system doesn't thread the plot Queue through.
    Re-add later if needed (PF already accepts an optional `queue`).
    """
    processes = launch_system([ParticleFilterAutoAimEngine])
    try:
        for p in processes:
            p.join()
    except KeyboardInterrupt:
        for p in processes:
            p.terminate()
            p.join()
