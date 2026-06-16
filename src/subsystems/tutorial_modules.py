from src.core.module import Module, real
from src.types.tutorial import CounterContext

class IncrementerModule(Module[CounterContext]):

    def __init__(self, context: CounterContext):
        super().__init__(
            name="incrementer",
            context=context,
            inputs=[],
            outputs=["number"],
        )
        self._count = 0
    
    @real()
    def _run(self):
        self._count += 1
        return self._count
    

class DoublerModule(Module[CounterContext]):

    def __init__(self, context: CounterContext):
        super().__init__(
            name="doubler",
            context=context,
            inputs=["number"],
            outputs=["doubled"],
        )
    
    @real()
    def _run(self, number):
        return number * 2