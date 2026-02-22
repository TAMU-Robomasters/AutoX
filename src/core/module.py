"""Base Module class for pipeline components."""

import inspect
from abc import ABC
from dataclasses import dataclass
from typing import Callable, Generic, List, Optional, TypeVar

from src.toolbox.globals import config


@dataclass
class Context:
    """Context is a simple data container that holds the state."""


T = TypeVar("T", bound="Context")  # type must inherit from Context


# ---------------------------------------------------------------------------
# Decorators for tagging run-method variants
# ---------------------------------------------------------------------------


def mock(fn: Callable) -> Callable:
    """Mark a method as the mock implementation for this module.

    There should be exactly one ``@mock`` method per module.  It runs
    when ``config.mock.enable`` is ``True``.

    Example::

        @mock
        def _run_mock(self, ctx):
            ...
    """
    fn._variant_kind = "mock"  # type: ignore[attr-defined]
    return fn


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

    def decorator(fn: Callable) -> Callable:
        fn._variant_kind = "real"  # type: ignore[attr-defined]
        fn._variant_requires = requires  # type: ignore[attr-defined]
        return fn

    return decorator


# ---------------------------------------------------------------------------
# Module base class
# ---------------------------------------------------------------------------


class Module(ABC, Generic[T]):
    """Modules are autonomous units used in pipelines.

    Subclasses define their behaviour by decorating methods with ``@real``
    and/or ``@mock``.  The base ``run()`` method dispatches automatically:

    * If ``config.mock.enable`` is ``True`` -> call the ``@mock`` method.
    * Otherwise -> call the ``@real`` method.

    Engines call ``module.run(ctx)`` -- that is the only public entry point.
    """

    def __init__(self, name: str, inputs: List[str], outputs: List[str]):
        """Initialize the module."""
        self._name = name
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

    # ------------------------------------------------------------------
    # run -- engines call this
    # ------------------------------------------------------------------
    def run(self, ctx: T) -> T:
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
            return self._mock_fn(ctx)

        # Pick a @real method. Future: match `requires` against hardware profile.
        if self._real_fns is None:
            raise RuntimeError(
                f"Module '{self._name}': no @real method was registered."
            )
        chosen = self._real_fns[
            0
        ]  # TODO: more sophisticated selection when multiple @real methods

        return chosen(ctx)

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
