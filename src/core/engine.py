"""Engines are systems that use modules to accomplish a specific goal.

They provide a high-level interface for the orchestrator to manage their status
and provide an easy way to swap out modules with their mocks during runtime.
"""

import inspect
from abc import ABC, abstractmethod
from dataclasses import dataclass, fields
from typing import Generic, List, Optional, Set, Type, TypeVar

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
        self.log = get_logger(type(self).__name__)
        self._validate_wiring()

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
            self._driver_handles[name] = driver_type.client(conn)

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
        configure_child_logging(self._log_queue)  # route this child's logs to the parent
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
