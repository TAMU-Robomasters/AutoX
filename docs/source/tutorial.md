# Tutorial: build your first engine

This is a hands-on, ~15-minute walkthrough. You'll build a toy engine with two
modules — one that counts up, one that doubles the count — and run it. It uses
only the core framework (no camera, GPU, or hardware), so it runs anywhere the
project installs.

By the end you'll have touched all four core nouns from {doc}`intro`: a
**Context**, two **Modules**, and an **Engine** (no **Driver** — this toy needs no
hardware).

```{tip}
Everything you create here is throwaway scaffolding. Delete the three files when
you're done — they're for learning, not part of the real system.
```

## Step 1 — define a Context

The context is the shared clipboard the modules read from and write to. Create
`src/types/tutorial.py`:

```python
from dataclasses import dataclass
from typing import Optional
from src.core.module import Context


@dataclass
class CounterContext(Context):
    number: Optional[int] = None
    doubled: Optional[int] = None
```

Two slots: `number` and `doubled`. Both start as `None`.

## Step 2 — write two modules

Create `src/subsystems/tutorial_modules.py`:

```python
from src.core.module import Module, real
from src.types.tutorial import CounterContext


class IncrementerModule(Module[CounterContext]):
    """Counts up by one every time it runs. Produces `number`."""

    def __init__(self, context: CounterContext):
        super().__init__(
            name="incrementer",
            context=context,
            inputs=[],            # needs nothing from the context
            outputs=["number"],   # writes ctx.number
        )
        self._count = 0

    @real()
    def _run(self):
        self._count += 1
        return self._count


class DoublerModule(Module[CounterContext]):
    """Reads `number`, writes `doubled`."""

    def __init__(self, context: CounterContext):
        super().__init__(
            name="doubler",
            context=context,
            inputs=["number"],    # reads ctx.number
            outputs=["doubled"],  # writes ctx.doubled
        )

    @real()
    def _run(self, number):
        return number * 2
```

Things to notice:

- `inputs` / `outputs` are just **strings** that must match field names on
  `CounterContext`.
- `@real()` marks "the real implementation." The base class reads `self.ctx.<name>`
  for each name in `inputs`, calls your method with those as positional arguments,
  and writes the return value(s) into the fields named in `outputs`.
- Nobody calls `_run` directly. The engine calls `module.run()`, which dispatches
  to the `@real` method (or a `@mock` one, if you wrote one and `config.mock.enable`
  is `True`).

```{note}
Want to see the mock path? Add a `@mock` method to a module and run with
`@mock.enable=True` (or a profile that sets it). The base class will call the
`@mock` version instead — with the engine and modules otherwise untouched. See
{doc}`module` for the decorator details.
```

## Step 3 — write the engine

The engine owns the context and the modules and runs them in a loop. Create
`src/engines/tutorial_counter_engine.py`:

```python
import time

from src.core.engine import Engine
from src.types.tutorial import CounterContext
from src.subsystems.tutorial_modules import IncrementerModule, DoublerModule


class CounterEngine(Engine[CounterContext]):
    def __init__(self):
        self.ctx = CounterContext()
        self.incrementer = IncrementerModule(self.ctx)
        self.doubler = DoublerModule(self.ctx)
        super().__init__(
            modules=[self.incrementer, self.doubler],
            context_type=CounterContext,
        )

    def initialize(self) -> None:
        pass  # nothing to set up — no camera / serial / GPU here

    def execute(self) -> None:
        self.incrementer.run()
        self.doubler.run()
        print(f"{self.ctx.number} doubled is {self.ctx.doubled}")
        time.sleep(1)


if __name__ == "__main__":
    engine = CounterEngine()
    engine.start()        # forks the child process, which runs initialize() then execute() in a loop
    try:
        engine.join()
    except KeyboardInterrupt:
        engine.stop()
```

A few things worth understanding here:

- The engine subclasses `Engine[CounterContext]`. The `[CounterContext]` is the
  context type, used for wiring validation (next section).
- `__init__` builds the context and modules and passes the module list +
  `context_type` to `super().__init__`. It does **no** hardware setup — that rule
  from {doc}`intro` applies even to a toy.
- `execute()` is one loop iteration: run each module in order, then (here) print
  and sleep. The base `Engine.run()` calls `execute()` repeatedly until the engine
  is stopped.
- This engine declares **no drivers**, so you can construct it and call `.start()`
  directly. (`Engine` is a `multiprocessing.Process`, so `start` / `join` / `stop`
  are process controls.) An engine that *did* declare `drivers` would instead be
  launched with `launch_system([...])` — see {doc}`intro`.

## Step 4 — run it

```sh
uv run python -m src.engines.tutorial_counter_engine
```

You should see, once per second, until you `Ctrl+C`:

```
1 doubled is 2
2 doubled is 4
3 doubled is 6
...
```

## Why it didn't blow up at construction time

When you called `super().__init__(modules=..., context_type=CounterContext)`, the
base `Engine` ran `_validate_wiring()`. It checks that:

1. every input/output name (`number`, `doubled`) is actually a field on
   `CounterContext`, and
2. every input (`number`) is produced as an output by *some* module
   (`IncrementerModule` produces it).

Try it: change `DoublerModule`'s input from `"number"` to `"nubmer"` and rerun.
Instead of a confusing failure deep in the loop, you get a clear `ValueError` at
startup naming the unsatisfied input. This validation is the main safety net the
framework gives you — it catches typos and missing producers before the loop ever
starts.

## What you just learned

You built, in order:

- a **Context** (`CounterContext`) — the shared clipboard,
- two **Modules** (`IncrementerModule`, `DoublerModule`) — each declaring its
  `inputs`/`outputs` and doing one focused thing,
- an **Engine** (`CounterEngine`) — owning the context and modules and running
  them in a loop as its own process.

That's the same shape every real engine in `src/engines/` follows — they just add
real modules, drivers for hardware, and a `@mock` path for laptop testing.

## Clean up

Delete the three files you created — they're scaffolding, not part of the system:

```sh
rm src/types/tutorial.py \
   src/subsystems/tutorial_modules.py \
   src/engines/tutorial_counter_engine.py
```

## Next steps

- Read a real engine: `src/engines/full_state_autoaim.py` follows this exact
  pattern with detection / estimation / ballistics modules and the camera + MCU
  drivers.
- {doc}`module` and {doc}`engine` — the deeper "why" behind each abstraction.
- {doc}`intro` "Current state vs. future direction" — what's wired up vs. planned
  (auto-mocking by hardware profile, config-selected engines, engine factories).
