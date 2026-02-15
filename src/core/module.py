"""Base Module class for pipeline components."""

from abc import ABC, abstractmethod
from typing import Generic, List, TypeVar

from src.core.context import Context

T = TypeVar("T", bound="Context")  # type must inherit from Context


class Module(ABC, Generic[T]):
    """Modules are autonomous units that are used in pipelines. And have specific input requirements."""

    def __init__(self, name: str, inputs: List[str], outputs: List[str]):
        """Initialize the module."""
        self._name = name
        self._inputs = inputs
        self._outputs = outputs

    @abstractmethod
    def run(self, ctx: T) -> T:
        """Process the given context and return an updated context."""

    @property  # ensure no setter
    def name(self) -> str:
        """Get the module name."""
        return self._name

    def get_inputs(self) -> List[str]:
        """Get module input requirements."""
        return self._inputs.copy()  # prevent modification of internal list

    def get_outputs(self) -> List[str]:
        """Get module output requirements."""
        return self._outputs.copy()  # prevent modification of internal list
