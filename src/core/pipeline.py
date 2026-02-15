"""Abstract base class for a processing pipeline."""

from abc import ABC
from typing import Generic, List, TypeVar

from src.core.context import Context
from src.core.module import Module

T = TypeVar("T", bound="Context")  # type must inherit from Context


class Pipeline(ABC, Generic[T]):
    """Abstract base class for a processing pipeline."""

    def __init__(self, modules: List[Module]):
        """Initialize the pipeline."""
        self._modules: List[Module] = modules

    def validate_pipeline(self, ctx: T):
        """Make sure each module is compatible with the next."""
        ctx_elements = set(ctx.__annotations__.keys())

        for i in reversed(range(len(self._modules) - 1)):
            current_module = self._modules[i]
            next_module = self._modules[i + 1]
            for requirement in next_module.get_inputs():
                assert requirement in ctx_elements, (
                    f"The input {requirement} from module {next_module.name} is not recognized. Check for typos."
                )
                assert requirement in current_module.get_outputs(), (
                    f"Module {next_module.name} requires {requirement} which is not in outputs of {current_module.name}"
                )

    def run(self, ctx: T):
        """Process the given context and return an updated context."""
        for module in self._modules:
            ctx = module.run(ctx)
