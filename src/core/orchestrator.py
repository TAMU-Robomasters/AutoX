"""Orchestrator: creates engines, wires shared queues, and starts everything."""

from multiprocessing import Queue

from src.toolbox.globals import config
from src.toolbox.logger import start_log_listener, stop_log_listener


def _link_names(engine_cls, single_attr: str, multi_attr: str) -> list:
    """Union of an engine's single-link name (if set) and its multi-link list.

    The single-link form (``publishes_queue`` / ``subscribes_queue``) is just the
    one-element case of the multi-link form (``publishes_queues`` /
    ``subscribes_queues``); both contribute names.
    """
    names = list(getattr(engine_cls, multi_attr, []) or [])
    single = getattr(engine_cls, single_attr, None)
    if single and single not in names:
        names.append(single)
    return names


def _wire_engine_queues(engine_classes: list) -> dict:
    """Build one shared queue per inter-engine pub/sub link name (plans/03).

    Reads each engine class's single- *and* multi-link declarations
    (``publishes_queue(s)`` / ``subscribes_queue(s)``). Two engines publishing the
    same name is a hard error (the global-namespace safety net). A subscribe with
    no publisher is allowed (logs nothing yet) -- the link is simply never fed.
    Returns ``{name: Queue}``.
    """
    publishers: dict = {}
    names: set = set()
    for engine_cls in engine_classes:
        for pub in _link_names(engine_cls, "publishes_queue", "publishes_queues"):
            if pub in publishers:
                raise RuntimeError(
                    f"Inter-engine queue {pub!r} is published by both "
                    f"{publishers[pub].__name__} and {engine_cls.__name__}; "
                    f"each link must have exactly one publisher."
                )
            publishers[pub] = engine_cls
            names.add(pub)
        for sub in _link_names(engine_cls, "subscribes_queue", "subscribes_queues"):
            names.add(sub)
    return {name: Queue() for name in names}


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

    # One shared driver per unique name across all engines, plus the list of
    # engine classes that consume each (so drivers can pre-allocate one resource
    # per consumer — e.g. McuDriver's per-consumer response queues).
    needed: dict = {}
    consumers_by_driver: dict = {}
    for engine_cls in engine_classes:
        for name, driver_type in getattr(engine_cls, "drivers", {}).items():
            needed[name] = driver_type
            consumers_by_driver.setdefault(name, []).append(engine_cls.__name__)

    registry: dict = {}
    processes: list = []
    for name, driver_type in needed.items():
        # parent-side transport allocation; consumers let the driver size
        # per-consumer resources (drivers that don't need it ignore the arg).
        conn = driver_type.provision(name, consumers_by_driver[name])
        registry[name] = conn
        driver = driver_type.from_conn(conn)
        driver.start()
        processes.append(driver)

    queue_registry = _wire_engine_queues(engine_classes)

    for engine_cls in engine_classes:
        engine = engine_cls(driver_registry=registry)
        engine._log_queue = log_queue  # inherited by the child at fork (see Engine.run)
        # Inter-engine pub/sub queues (inherited by the child at fork too). Set the
        # single-link handles for the name=None path, and the name->queue dicts for
        # the multi-link path; both include every link this engine declares.
        pub = getattr(engine_cls, "publishes_queue", None)
        sub = getattr(engine_cls, "subscribes_queue", None)
        if pub:
            engine._publish_q = queue_registry[pub]
        if sub:
            engine._subscribe_q = queue_registry[sub]
        engine._publish_qs = {
            n: queue_registry[n]
            for n in _link_names(engine_cls, "publishes_queue", "publishes_queues")
        }
        engine._subscribe_qs = {
            n: queue_registry[n]
            for n in _link_names(engine_cls, "subscribes_queue", "subscribes_queues")
        }
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

    When ``config.circlet.enable`` is set (the ``@CIRCLET`` profile) the circlet
    ring engine is launched alongside auto-aim; their detections link is wired
    automatically by ``launch_system`` (publishes/subscribes ``circlet_detections``).
    Engines are imported lazily so this module stays importable without the
    detector/mrcal stack (e.g. for the queue-wiring unit tests).
    """
    from src.engines.full_state_autoaim import FullStateAutoAimEngine

    engine_classes: list = [FullStateAutoAimEngine]
    circlet_cfg = getattr(config, "circlet", None)
    if circlet_cfg is not None and getattr(circlet_cfg, "enable", False):
        from src.engines.circlet import CircletEngine

        engine_classes.append(CircletEngine)

    processes = launch_system(engine_classes)
    try:
        for p in processes:
            p.join()
    except KeyboardInterrupt:
        for p in processes:
            p.terminate()
            p.join()
    finally:
        stop_log_listener()  # flush any queued records before exit
