from typing import Set

def module(requires: Set[str], produces: Set[str]):
    """Decorator to explicitly declare module interface."""
    def decorator(cls):
        cls.requires = requires
        cls.produces = produces
        return cls
    return decorator

@module(requires={"input1", "input2"}, produces={"output1"})
class foo():
    def __init__(self):
        pass

    def run(self, ctx):
        # Process inputs and produce outputs
        ctx.output1 = ctx.input1 + ctx.input2