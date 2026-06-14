# Engines

## What an engine is

An engine is the process-level unit of work in AutoX. If modules are the
algorithmic building blocks, an engine is the thing that owns those blocks and runs
them as a live system. The base class (`src/core/engine.py`) subclasses
`multiprocessing.Process`, so **each engine runs as its own OS process**.

An engine owns:

- one {doc}`context` (the shared state threaded through its modules),
- an ordered list of {doc}`modules <module>`,
- its process lifecycle, and
- any process-local resources (IPC endpoints, the driver handles it asked for).

## The lifecycle

You construct an engine, `start()` it, and the rest happens in the child process:

1. construct the engine object (in the parent)
2. `start()` forks the process
3. the child calls `initialize()` once
4. the child calls `execute()` repeatedly (one iteration of work each call)
5. `stop()` signals shutdown and joins

You implement `initialize()` and `execute()`; the base class provides `run()`
(the loop), `stop()`, and startup checks.

## Why `initialize()` is separate from `__init__()`

`__init__()` runs in the **parent** process, before the fork. Opening a camera,
serial port, IPC node, or display window there means the resource is created in the
parent and then copied into the child — which is fragile or outright broken.
`initialize()` runs **inside the child**, so that's where all such setup belongs.
Modules have their own `initialize()` for the same reason (see {doc}`module`); the
engine calls each module's after its own. This is the single most common
multiprocessing mistake in the codebase — the {doc}`intro` calls it out as "the one
rule that saves the most debugging time."

## What the base class gives you

### Wiring validation

`Engine.__init__` calls `_validate_wiring()` before anything runs: every module
input/output name must be a real field on the context dataclass, and every input
must be produced by some module's output. A typo'd field name raises `ValueError`
at construction, not mid-run. (See {doc}`context`.)

### A pipeline log on startup

Each engine logs its module → inputs → outputs table at INFO when it starts, so you
can see the wired pipeline in the logs.

### Driver handles

An engine declares the shared hardware it needs as a class attribute:

```python
class FullStateAutoAimEngine(Engine[FullStateAutoAimContext]):
    drivers = {"frames": CameraDriver, "mcu": McuDriver}
```

A {doc}`driver <intro>` is hardware that only one process may own (the camera, the
MCU serial link). The engine doesn't open those itself — it receives a client
handle for each and reaches it with `self.driver("frames")`. Engines that declare
`drivers` **must** be launched through `launch_system` (below); constructing one and
calling `.start()` directly raises a `RuntimeError`.

## How engines are launched

An engine **without** drivers can be constructed and started directly — this is
what the {doc}`tutorial` does:

```python
engine = CounterEngine()
engine.start()
engine.join()
```

An engine **with** drivers goes through the orchestrator, which spawns one shared
process per declared driver, builds the registry, and wires the handles in before
the engine's `initialize()` runs:

```python
from src.core.orchestrator import launch_system
processes = launch_system([FullStateAutoAimEngine])
```

`launch_system` is the working driver-aware factory today. The surrounding
watchdog/supervision behavior (liveness, restart, config-driven engine selection)
is the planned direction described in {doc}`intro` — `src/core/orchestrator.py`
is where it's headed, not where it fully is yet.

## Why the engine, not the modules, owns the control flow

The engine calls `module.run()` directly inside `execute()` rather than handing the
whole pipeline off to a generic runner. That's deliberate: it leaves room for
cross-module logic and side effects that don't belong in any single module — frame
pacing, "skip targeting if there are no panels," talking to the MCU, updating the
display. The production engine
(`src/engines/full_state_autoaim.py`, see {doc}`full_state_autoaim_engine`) is the
clearest case: it runs a per-robot state machine that chooses *which* modules and
*which* ballistics run each frame, and owns all the disk and MCU I/O, while the
modules stay reusable and stateless about that control flow.

## Other engines in the repo

`src/engines/` has smaller engines that are useful as references for the pattern:

- `CountEngine` (`count.py`) has **no modules at all** — it's the minimal example
  that an engine is about running a managed process loop, not only about driving a
  module pipeline.
- `detection_test_engine*.py`, `plot_engine.py`, `pf_test_engine.py`, and the
  archived `archive_full_state_autoaim.py` are working references for different
  pieces.

```{note}
`autoaim.py` contains `SimpleAutoAimEngine`, but it's **commented out** — it
predates the full-state pipeline and uses the older `AutoAimContext` +
iceoryx2 IPC. Read it for historical context, not as a current entry point. The
live engine is `FullStateAutoAimEngine`.
```
