import time

from src.core.engine import Engine
from src.types.tutorial import CounterContext
from src.subsystems.tutorial_modules import IncrementerModule, DoublerModule

class CounterEngine(Engine[CounterContext]):
    def __init__(self, driver_registry=None):
        self.ctx = CounterContext()
        self.incrementer = IncrementerModule(self.ctx)
        self.doubler = DoublerModule(self.ctx)
        super().__init__(
            modules=[self.incrementer, self.doubler],
            context_type=CounterContext,
            driver_registry=driver_registry,
        )

    def initialize(self) -> None:
        pass #bc there's nothing to set up here, no camera/serial/GPU

    def execute(self) -> None:
        self.incrementer.run()
        self.doubler.run()
        print(f"{self.ctx.number} doubled is {self.ctx.doubled}")
        time.sleep(1)

if __name__ == "__main__":
    engine = CounterEngine()
    engine.start()
    try:
        engine.join()
    except KeyboardInterrupt:
        engine.stop()
