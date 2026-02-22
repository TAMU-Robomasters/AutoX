"""Engines are systems that use modules to accomplish a specific goal.

They provide a high-level interface for the orchestrator to manage their status
and provide an easy way to swap out modules with their mocks during runtime.
"""

import inspect
from abc import ABC, abstractmethod
from dataclasses import dataclass, fields
from typing import Generic, List, Set, Type, TypeVar

from src.core.module import Context, Module

T = TypeVar("T", bound="Context")


class Engine(ABC, Generic[T]):
    """Abstract base class for an engine.

    An engine owns an ordered list of :class:`Module` instances and is
    responsible for:

    * **Wiring validation** – every module input must be satisfied by an
      output produced by a another module. Outputs that are never consumed are acceptable.

    Subclasses must implement :meth:`start` which defines how the engine
    actually drives its modules (single pass, loop, async, state machine, threaded, etc.).
    """

    def __init__(self, modules: List[Module], context_type: Type[T]):
        """Initialize the engine.

        Args:
            modules: Ordered list of modules that are used by this engine.
            context_type: The type of the initial context that will be used
                            by the engine's modules.  This is used for
                            validating that inputs and outputs from modules are
                            actually valid and present in the context.
        """
        self._modules: List[Module] = modules
        self._initial_context_keys: set[str] = {f.name for f in fields(context_type)}
        self._validate_wiring()

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

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    @abstractmethod
    def start(self):
        """Execute the engine's processing logic against *ctx*."""
