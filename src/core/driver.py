import functools
from multiprocessing import Process as _Process
from abc import ABC, abstractmethod
from typing import Callable, Optional

#NOTE: kinda just playing around with this abstraction not sure how useful it will be

def mock(fn: Callable) -> Callable:
    """Mark a method as a mock implementation.

    There should be exactly one ``@mock`` method for ``initialize`` and ``execute.  
    It runs when ``config.mock.enable`` is ``True``.

    Example::

        @mock
        def _initialize_mock(self):
            ...
        
        @mock
        def _execute_mock(self):
            ...
    """

    @functools.wraps(fn)
    def wrapper(self):
        return fn(self)

    wrapper._variant_kind = "mock"  # type: ignore[attr-defined]
    return wrapper


def real(fn: Callable) -> Callable:
    """Mark a method as a real implementation.

    There should be exactly one ``@real`` method for ``initialize`` and ``execute.  
    It runs when ``config.mock.enable`` is ``False``.

    Example::

        @real
        def _initialize_real(self):
            ...
        
        @real
        def _execute_real(self):
            ...
    """

    @functools.wraps(fn)
    def wrapper(self):
        return fn(self)

    wrapper._variant_kind = "real"  # type: ignore[attr-defined]
    return wrapper


class Driver(_Process, ABC):
    """Drivers are designed to encapsulate a piece of hardware that can only be managed by a single process.

    Common examples are:
    - Cameras: Most camera APIs don't allow you to open and then grab the frame in two different processes
    - MCU: The MCU (micro controller unit) or Devboard will need to communicate with different processes but the serial bus (UART)
      can only be managed by a single process (at least if you're using pyserial).
    """

    def __init__(self, requires: str):
        """Initialize the engine.

        Args:
            requires: This will be used for future profile-based hardware matching. If the device doesn't match the requirements, 
            then the mock implementation of this driver will automatically be used. 
        """
        super().__init__()
        

    @abstractmethod
    def initialize(self):
        """Put initialization stuff here. We want to avoid using __init__  because of multiprocessing.

        If you put stuff in __init__, it will run in the parent process and might create bugs when trying
        to copy the memory into the child process if the stuff you're copying over is multi-threaded.
        """

    def run(self):
        """This is what will be called when you do engine.start().

        Main loop that runs until the engine is stopped.
        """
        self.active = True
        self.initialize()
        while self.active:
            self.execute()

    @abstractmethod
    def execute(self):
        """The main body of an engine. Called repeatedly."""

    def stop(self):
        """Signal the engine to stop and wait for it to finish."""
        self.active = False
        self.join()
