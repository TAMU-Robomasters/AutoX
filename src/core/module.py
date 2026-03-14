"""Base Module class for pipeline components."""

import inspect
from abc import ABC
from dataclasses import dataclass
from typing import Callable, Generic, List, Optional, TypeVar, Tuple

from src.toolbox.globals import config

# TODO: see if I can check if the types are in the same order as the inputs and outputs, by checking the type


@dataclass
class Context:
    """Context is a simple data container that holds the state."""


T = TypeVar("T", bound="Context")  # type must inherit from Context


# ---------------------------------------------------------------------------
# Decorators for tagging run-method variants
# ---------------------------------------------------------------------------

#TODO decide if we want ctx to be optional

def mock(fn: Callable) -> Callable:
    """Mark a method as the mock implementation for this module.

    There should be exactly one ``@mock`` method per module.  It runs
    when ``config.mock.enable`` is ``True``.

    Example::

        @mock
        def _run_mock(self, ctx):
            ...
    """
    import functools

    @functools.wraps(fn)
    def wrapper(self):
        args = [getattr(self.ctx, input) for input in self._inputs]
        outputs = fn(self, *args)
        if not isinstance(outputs, tuple):
            outputs = (outputs,)
        for output, value in zip(self._outputs, outputs):
            setattr(self.ctx, output, value)
        return self.ctx

    wrapper._variant_kind = "mock"  # type: ignore[attr-defined]
    return wrapper


def real(requires: Optional[str] = None) -> Callable:
    """Mark a method as a real implementation for this module.

    Args:
        requires: Optional hardware dependency (e.g. ``"camera"``).
                  Stored for future profile-based hardware matching.
                  A ``@real()`` with no *requires* means "no special
                  hardware needed -- always eligible".

    Multiple ``@real`` methods are allowed on a single module when they
    have different *requires* values (future: the module will pick the
    one whose requirements are satisfied by the active hardware profile).

    Example::

        @real(requires="camera")
        def _run_detect(self, ctx):
            ...
    """

    def inner(fn: Callable) -> Callable:
        import functools

        @functools.wraps(fn)
        def wrapper(self):
            args = [getattr(self.ctx, input) for input in self._inputs]
            outputs = fn(self, *args)
            # if outputs is not an iterable (like a tuple), we should make it one
            # to match zip(self._outputs, outputs) unless len(outputs) == 1
            if not isinstance(outputs, tuple):
                outputs = (outputs,)
            for output, value in zip(self._outputs, outputs):
                setattr(self.ctx, output, value)
            return self.ctx

        wrapper._variant_kind = "real"  # type: ignore[attr-defined]
        wrapper._variant_requires = requires  # type: ignore[attr-defined]
        return wrapper

    return inner


# ---------------------------------------------------------------------------
# Module base class
# ---------------------------------------------------------------------------


# TODO: check the parameter names on the methods and the inputs to assert they match
class Module(ABC, Generic[T]):
    """Modules are autonomous units used in pipelines.

    Subclasses define their behaviour by decorating methods with ``@real``
    and/or ``@mock``.  The base ``run()`` method dispatches automatically:

    * If ``config.mock.enable`` is ``True`` -> call the ``@mock`` method.
    * Otherwise -> call the ``@real`` method.

    Engines call ``module.run(ctx)`` -- that is the only public entry point.
    """

    def __init__(self, name: str, context: T, inputs: List[str], outputs: List[str]):
        """Initialize the module."""
        self._name = name
        self.ctx = context
        self._inputs = inputs
        self._outputs = outputs
        self._mock_fn: Optional[Callable] = None
        self._real_fns: Optional[List[Callable]] = None

        # Discover decorated variant methods on this instance's class.
        mock: List = inspect.getmembers(
            self, lambda m: getattr(m, "_variant_kind", None) == "mock"
        )
        if mock:
            self._mock_fn = mock[0][1]

        reals = inspect.getmembers(
            self, lambda m: getattr(m, "_variant_kind", None) == "real"
        )
        if reals:
            self._real_fns = [r[1] for r in reals]

        self._run_method: Callable = self._select_run_method()

    # ------------------------------------------------------------------
    # run -- engines call this
    # ------------------------------------------------------------------

    def _select_run_method(self) -> Callable:
        """Dispatch to the appropriate @real or @mock method.

        * ``config.mock.enable`` is ``True``  -> ``@mock``
        * ``config.mock.enable`` is ``False`` -> ``@real``
        """
        use_mock = False
        if getattr(config, "mock", None) is not None and config.mock.enable:
            use_mock = True

        if use_mock:
            if self._mock_fn is None:
                raise RuntimeError(
                    f"Module '{self._name}': mock is required "
                    f"(config.mock.enable is True) but no @mock method "
                    f"was registered."
                )
            return self._mock_fn

        # Pick a @real method. Future: match `requires` against hardware profile.
        if self._real_fns is None:
            raise RuntimeError(
                f"Module '{self._name}': no @real method was registered."
            )
        chosen = self._real_fns[
            0
        ]  # TODO: more sophisticated selection when multiple @real methods

        return chosen

    def remap_inputs(self, name: str, old_input: List[str], new_input: List[str]):
        assert len(old_input) == len(new_input), (
            "Input remapping requires lists of the same length."
        )
        for old, new in zip(old_input, new_input):
            self._inputs = [new if i == old else i for i in self._inputs]

    def remap_outputs(self, old_output: List[str], new_output: List[str]):
        assert len(old_output) == len(new_output), (
            "Output remapping requires lists of the same length."
        )
        for old, new in zip(old_output, new_output):
            self._outputs = [new if o == old else o for o in self._outputs]

    def run(self) -> T:
        """Run the module on the given context."""
        return self._run_method()

    # ------------------------------------------------------------------
    # Properties / accessors
    # ------------------------------------------------------------------

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
