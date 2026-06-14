# Modules

## What a module is

A module is the smallest unit of work in AutoX. It's a subclass of `Module[T]`
(base in `src/core/module.py`) that:

- reads named values from a shared {doc}`context` (`inputs`),
- does one focused piece of computation, and
- writes named results back to that context (`outputs`).

The engine only ever calls `module.run()`. It doesn't know how the module works
internally — only what fields it consumes and produces. `run()` pulls the declared
inputs off the context by name, calls the selected implementation, and writes the
returned value(s) back to the declared output fields.

```python
class ExampleModule(Module[SomeContext]):
    def __init__(self, context: SomeContext):
        super().__init__(
            name="ExampleModule",
            context=context,
            inputs=["some_input"],
            outputs=["some_output"],
        )

    @real()
    def _run_real(self, some_input):
        return transform(some_input)
```

## Why split work into modules

The vision/auto-aim stack is naturally a sequence of stages — detect, classify,
estimate state, pick a target, solve ballistics, talk to the MCU. Breaking those
into modules means each stage can be swapped, reused, or tested on its own without
touching the engine loop around it. The chain in the production engine, for
example, runs roughly:

- `CppDetectorModule` (or `ClassicalDetectorModule`) — `inputs=[]`,
  `outputs=["panels"]`. An entry-point stage: it grabs a frame and produces
  detected armor panels.
- `RobotClassificationModule` — consumes `panels`, tags each with which robot it
  belongs to.
- `TargetingModule` — picks the `target_robot`.
- estimation + ballistics modules — turn the target into a kinematic `estimate`
  and then a firing `solution`.

Each stage exposes the same small contract (`context in → compute → context out`),
even the ones that touch hardware or IPC internally.

## The `@real` / `@mock` split

Every module can carry two implementations:

- `@real(...)` — the production version (real camera/GPU/hardware).
- `@mock` — a simulated version that returns canned data.

The base class chooses which to run in `_select_run_method()`, based on the single
config flag `config.mock.enable` (from `info.yaml`). When it's true, the `@mock`
method runs; otherwise the first `@real` method runs. Because engines only call
`module.run()`, **no call site changes** when you switch the whole system between
real and mock — that's what lets the full pipeline run on a laptop with nothing
plugged in.

```python
class DetectorModule(Module[FullStateAutoAimContext]):
    @real(requires="camera")
    def _run_real(self):
        ...                       # grab a frame, detect panels
        return panels

    @mock
    def _run_mock(self):
        return [fake_panel()]     # deterministic, no hardware
```

```{note}
`requires="camera"` is **recorded but not yet acted on**. Today
`_select_run_method` ignores it and always picks the first `@real` (there's a
`TODO` on that line); the only switch actually honored is the global
`config.mock.enable`. The intent — falling back to `@mock` per-module when a board
lacks the required hardware — is described under "Current state vs. future
direction" in {doc}`intro`.
```

## Per-module setup: `Module.initialize()`

`Module.initialize()` is a one-time, per-module setup hook that runs **in the child
process** — `Engine.run()` calls it for every module after the engine's own
`initialize()`. Put anything that must not be created in the parent process
(hardware handles, native libraries, IPC) here, **not** in `__init__` and not
lazily on the first `run()`. See the multiprocessing rule in {doc}`intro` and
{doc}`engine` for why.

## Reuse: input/output remapping

`remap_inputs()` and `remap_outputs()` let the same module class be wired into a
context whose field names differ from the ones the module declares, so a stage can
be reused across pipelines without editing its implementation.

## Tradeoffs

Field names are plain strings, so typos matter — but `_validate_wiring()` (see
{doc}`context`) catches them at construction, before the loop runs. The upside is
small, swappable stages with explicit data dependencies and an easy mock/real
switch.
