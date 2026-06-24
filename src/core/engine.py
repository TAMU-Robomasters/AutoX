"""Engines are systems that use modules to accomplish a specific goal.

They provide a high-level interface for the orchestrator to manage their status
and provide an easy way to swap out modules with their mocks during runtime.
"""

import inspect
import queue as _queue
from abc import ABC, abstractmethod
from dataclasses import dataclass, fields
from typing import Any, Generic, List, Optional, Set, Type, TypeVar

from src.core.module import Context, Module
from src.toolbox.logger import configure_child_logging, get_logger

from multiprocessing import Process as _Process

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
    publishes_queue: Optional[str] = None
    subscribes_queue: Optional[str] = None

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
        self._publish_q = None
        self._subscribe_q = None
        self.log = get_logger(type(self).__name__)
        self._validate_wiring()

    # ------------------------------------------------------------------
    # Inter-engine pub/sub (interim queue link -- plans/03)
    # ------------------------------------------------------------------

    def publish(self, message: Any) -> None:
        """Publish ``message`` on this engine's outbound queue (last-value, non-blocking).

        Drains any unread prior message first so the queue holds ~one item and a
        slow consumer always reads the newest -- the same last-value semantics as
        the camera ring buffer. No-op if this engine declares no ``publishes_queue``.
        """
        if self._publish_q is None:
            return
        try:
            while True:
                self._publish_q.get_nowait()
        except _queue.Empty:
            pass
        self._publish_q.put(message)

    def latest_subscribed(self) -> Optional[Any]:
        """Return the newest message on the inbound queue (draining older), or None.

        No-op (returns None) if this engine declares no ``subscribes_queue``.
        """
        if self._subscribe_q is None:
            return None
        message = None
        try:
            while True:
                message = self._subscribe_q.get_nowait()
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
