from multiprocessing import Process as _Process
from abc import abstractmethod, ABC

class Process(_Process, ABC):
    def __init__(self):
        super().__init__()
        self.active = True

    def run(self):
        """Name is confusing but this what will be called when you do process.start() 
        Main loop that runs until the process is stopped."""
        while self.active:
            self.execute()

    @abstractmethod
    def execute(self):
        """The main body of a process. Called repeatedly."""
        
    
    def stop(self):
        """Signal the process to stop and wait for it to finish."""
        self.active = False
        self.join()