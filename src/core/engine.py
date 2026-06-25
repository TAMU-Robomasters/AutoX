"""Engines are systems that use modules to accomplish a specific goal.

They provide a high-level interface for the orchestrator to manage their status
and provide an easy way to swap out modules with their mocks during runtime.
"""

import inspect
import queue as _queue
from abc import ABC, abstractmethod
from dataclasses import dataclass, fields
from multiprocessing import Process as _Process
from typing import Any, Generic, List, Optional, Set, Type, TypeVar

from src.core.module import Context, Module
from src.toolbox.logger import configure_child_logging, get_logger

T = TypeVar("T", bound="Context")


class Engine(_Process, ABC, Generic[T]):
    """Abstract base class for an engine.

    An engine owns an ordered list of :class:`Module` instances and is
    responsible for:

    * **Wiring validation** – every module input must be satisfied by an
      output produced by a another module. Outputs that are never consumed are acceptable.

    Subclasses must implement :meth:`initialize` and :meth:`execute` instead of `__init__`
    to avoid running initialization code in the parent process before multiprocessing fork/spawn.
    """

    #: Drivers this engine consumes, ``{name: DriverType}``. The orchestrator
    #: factory provisions one shared driver process per name and passes a
    #: ``driver_registry`` so the engine can build a client handle in its child
    #: process. Override in subclasses; default = no drivers.
    drivers: dict = {}

    #: Inter-engine pub/sub link names (the ``plans/03`` interim shortcut over a
    #: pickled ``multiprocessing.Queue``). ``launch_system`` creates one shared
    #: queue per unique name and injects it before ``start()`` (so the child
    #: inherits it at fork). Two engines publishing the same name is a hard
    #: startup error. Use :meth:`publish` / :meth:`latest_subscribed`.
    #:
    #: ``publishes_queue`` / ``subscribes_queue`` are the **single-link** form (one
    #: name each); :meth:`publish` / :meth:`latest_subscribed` with no ``name`` use
    #: them. ``publishes_queues`` / ``subscribes_queues`` are the **multi-link**
    #: form (plan 09): an engine on several links (e.g. AutoAim subscribing to both
    #: ``circlet_detections`` and ``engage_directive``) lists them and addresses
    #: each by ``name``. The two forms compose — a name from either is reachable by
    #: ``name`` — so the single-link API is just the one-element case.
    publishes_queue: Optional[str] = None
    subscribes_queue: Optional[str] = None
    publishes_queues: List[str] = []
    subscribes_queues: List[str] = []

    def __init__(
        self,
        modules: List[Module],
        context_type: Type[T],
        driver_registry: Optional[dict] = None,
    ):
        """Initialize the engine.

        Args:
            modules: Ordered list of modules that are used by this engine.
            context_type: The type of the initial context that will be used
                            by the engine's modules.  This is used for
                            validating that inputs and outputs from modules are
                            actually valid and present in the context.
            driver_registry: ``{name: conn}`` from the orchestrator factory,
                            where ``conn`` is whatever the driver's
                            ``provision()`` returned (e.g. an iceoryx2 service
                            name or a queue pair). Used to build client handles.
        """
        super().__init__()
        self._modules: List[Module] = modules
        self._initial_context_keys: set[str] = {f.name for f in fields(context_type)}
        self._driver_registry: dict = driver_registry or {}
        self._driver_handles: dict = {}
        #: Shared multiprocess log queue, set by orchestrator.launch_system
        #: after construction (None = log straight to console, e.g. in tests).
        self._log_queue = None
        #: Inter-engine queues, set by launch_system after construction (None =
        #: no link, e.g. when an engine runs standalone or in a unit test).
        #: ``_publish_q`` / ``_subscribe_q`` back the single-link (name=None) path;
        #: ``_publish_qs`` / ``_subscribe_qs`` map link name -> queue for the
        #: multi-link path. launch_system populates the dicts with *all* this
        #: engine's links (single + multi) so any link is reachable by name.
        self._publish_q = None
        self._subscribe_q = None
        self._publish_qs: dict = {}
        self._subscribe_qs: dict = {}
        self.log = get_logger(type(self).__name__)
        self._validate_wiring()

    # ------------------------------------------------------------------
    # Inter-engine pub/sub (interim queue link -- plans/03)
    # ------------------------------------------------------------------

    def _resolve_link(self, name: Optional[str], single, multi: dict):
        """Pick the queue for ``name`` (or the implicit single link).

        Explicit ``name`` -> ``multi[name]``; ``None`` -> the single-link queue,
        else the sole multi link if there's exactly one. Returns ``None`` (callers
        no-op) when nothing matches.
        """
        if name is not None:
            return multi.get(name)
        if single is not None:
            return single
        return next(iter(multi.values())) if len(multi) == 1 else None

    def publish(self, message: Any, name: Optional[str] = None) -> None:
        """Publish ``message`` on a link (last-value, non-blocking).

        ``name`` selects a multi-link (``publishes_queues``); omit it for the
        single ``publishes_queue``. Drains any unread prior message first so the
        queue holds ~one item and a slow consumer always reads the newest -- the
        same last-value semantics as the camera ring buffer. No-op if the engine
        declares no matching link.
        """
        q = self._resolve_link(name, self._publish_q, self._publish_qs)
        if q is None:
            return
        try:
            while True:
                q.get_nowait()
        except _queue.Empty:
            pass
        q.put(message)

    def latest_subscribed(self, name: Optional[str] = None) -> Optional[Any]:
        """Return the newest message on a link (draining older), or None.

        ``name`` selects a multi-link (``subscribes_queues``); omit it for the
        single ``subscribes_queue``. No-op (returns None) if the engine declares no
        matching link.
        """
        q = self._resolve_link(name, self._subscribe_q, self._subscribe_qs)
        if q is None:
            return None
        message = None
        try:
            while True:
                message = q.get_nowait()
        except _queue.Empty:
            pass
        return message

    @abstractmethod
    def initialize(self):
        """Put initialization stuff here. We don't want to use __init__  because of multiprocessing.

        If you put stuff in __init__, it will run in the parent process and might create issues when trying
        to copy the memory into the child process.
        """

    def _build_driver_handles(self):
        """Create a client handle for each declared driver (child process)."""
        for name, driver_type in type(self).drivers.items():
            if name not in self._driver_registry:
                raise RuntimeError(
                    f"Engine '{type(self).__name__}' declares driver '{name}' but no "
                    f"driver registry was provided. Launch engines that declare "
                    f"`drivers` via `orchestrator.launch_system([...])`, not by "
                    f"constructing and .start()ing them directly."
                )
            conn = self._driver_registry[name]
            # Pass this engine's identity so drivers that fan out to multiple
            # concurrent consumers (e.g. McuDriver's per-consumer response queues)
            # can route replies back to the right client. Drivers that don't need
            # it ignore the argument.
            self._driver_handles[name] = driver_type.client(conn, type(self).__name__)

    def driver(self, name: str):
        """Return the client handle for a declared driver (built in run())."""
        return self._driver_handles[name]

    def run(self):
        """This is what will be called when you do engine.start().

        Main loop that runs until the engine is stopped. Everything here runs in
        the child process, so driver handles and module resources are built here
        (never in __init__, which runs in the parent before fork/spawn).
        """
        self.active = True
        configure_child_logging(
            self._log_queue
        )  # route this child's logs to the parent
        self._log_pipeline()
        self._build_driver_handles()  # driver clients, before initialize() can use them
        self.initialize()
        for module in self._modules:
            module.initialize()  # one-time per-module child-process setup
        while self.active:
            self.execute()
            self.update()

    @abstractmethod
    def execute(self):
        """The main body of an engine. Called repeatedly."""

    def update(self):
        """Runs every loop after execute. Useful for updating the display.

        Override if needed, otherwise does nothing.
        """
        pass

    def stop(self):
        """Signal the engine to stop and wait for it to finish."""
        self.active = False
        self.join()

    def _log_pipeline(self) -> None:
        """Log the module -> inputs -> outputs table once at startup.

        Derived from the modules' own declarations, so it can't go stale --
        a free pipeline 'diagram' for anyone reading the logs.
        """
        if not self.log.isEnabledFor(20):  # logging.INFO
            return
        lines = [
            f"  {m.name}: ({', '.join(m.get_inputs()) or '-'}) -> ({', '.join(m.get_outputs()) or '-'})"
            for m in self._modules
        ]
        self.log.info("pipeline:\n%s", "\n".join(lines))

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_wiring(self) -> None:
        """Verify that every module input is provided by a prior output.

        Raises:
            ValueError: If an input is not satisfied.
        """
        available: Set[str] = set(self._initial_context_keys)
        if not available:  # context is empty
            # TODO: warn users
            return

        outputs = {
            output for module in self._modules for output in module.get_outputs()
        }
        inputs = {input for module in self._modules for input in module.get_inputs()}
        missing = (outputs | inputs) - available
        if missing:
            raise ValueError(
                f"The following inputs and/or outputs are not in the context: {missing}."
                f"Fix potential typo or add the input or output to the appropriate context"
                f"Inputs and outputs in Context: {sorted(available)}"
            )

        input_missing_output = inputs - outputs
        if input_missing_output:
            raise ValueError(
                f"The following inputs are required by modules but not produced by any module: {input_missing_output}."
                f"Fix potential typo or add a module that produces the missing output"
            )

    # ------------------------------------------------------------------
    # Module access
    # ------------------------------------------------------------------

    @property
    def modules(self) -> List[Module]:
        """Return the ordered list of modules (read-only copy)."""
        return list(self._modules)

    def get_module(self, name: str) -> Module:
        """Look up a module by name.

        Raises:
            KeyError: If no module with that name exists.
        """
        for m in self._modules:
            if m.name == name:
                return m
        raise KeyError(
            f"No module named '{name}' in engine. "
            f"Available: {[m.name for m in self._modules]}"
        )
