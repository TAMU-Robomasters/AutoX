from abc import ABC, abstractmethod  # noqa: D100



class Process(_Process, ABC):
    """Base class for a process. Subclass this and implement *initialize* and *execute*."""

    @abstractmethod
    def initialize(self):
        """Put initialization stuff here. We don't want to use __init__  because of multiprocessing.

        If you put stuff in __init__, it will run in the parent process and might create issues when trying
        to copy the memory into the child process.
        """

    def run(self):
        """This is what will be called when you do process.start().

        Main loop that runs until the process is stopped.
        """
        self.active = True
        self.initialize()
        while self.active:
            self.execute()

    @abstractmethod
    def execute(self):
        """The main body of a process. Called repeatedly."""

    def stop(self):
        """Signal the process to stop and wait for it to finish."""
        self.active = False
        self.join()
