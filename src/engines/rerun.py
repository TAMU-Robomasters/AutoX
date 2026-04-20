from src.core.engine import Engine
from src.types.autoaim import AutoAimContext
from src.subsystems.visualization.rerun_model import Simulation
from src.subsystems.selection import SelectingWith3DModule
from src.subsystems.vision import ClassicalDetectorModule
from src.subsystems.display import display
import random

class RerunEngine(Engine[AutoAimContext]):
    """Use classical cv to find panels and 3D info to select the best target."""

    def __init__(self):
        self.ctx = AutoAimContext()
        self.simulate = Simulation(self.ctx)
        self.detection = ClassicalDetectorModule(self.ctx)
        super().__init__(
            modules=[
                self.detection,
                self.simulate
            ],
            context_type=AutoAimContext,
        )

    def initialize(self):  # noqa: D102
        self.simulate.init_stuff()
        pass

    def execute(self):  # noqa: D102
        self.detection.run(self.ctx)


        self.simulate.run(self.ctx)
    def end(self):
        self.simulate.ending()



 