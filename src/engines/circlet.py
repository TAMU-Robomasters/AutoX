"""Circlet camera engine"""

from src.core.engine import Engine
from src.core.process import Process
from src.subsystems.display import display
from src.subsystems.selection import SelectingWith3DModule
from src.subsystems.vision import ClassicalDetectorModule
from src.types.autoaim import AutoAimContext
from src.types.ipc import ImageMessage, LogMessage

class CircletProcess(Process):
    def initialize(self):  # noqa: D102
        self.ctx = AutoAimContext()
        
    def execute(self):  # noqa: D102
        ClassicalDetectorModule().run(self.ctx)

        display.show_windows()
    
class CircletEngine(Engine[AutoAimContext]):
    """Run Autoaim processes on each circlet camera in series"""
    
    def __init__(self):
        super().__init__(
            modules=[
                ClassicalDetectorModule(),
            ],
            context_type=AutoAimContext
        )
        self.circlet = CircletProcess()
        
    def start(self):
        self.circlet.start()