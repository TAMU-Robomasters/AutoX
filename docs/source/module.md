# Modules

## What a module is

A module is the smallest reusable unit of work in AutoX.

At the code level, a module is a subclass of `Module[T]` that:

- reads named values from a shared `Context`
- performs one focused piece of logic
- writes named results back into that same `Context`

The base implementation in `src/core/module.py` makes that contract explicit:

- each module declares `inputs`
- each module declares `outputs`
- the engine only calls `module.run()`

That means the engine does not need to know how a module works internally. It only needs to know what data the module consumes and what data it produces.

## Why this abstraction exists

This layer exists to solve a few problems that show up constantly in robotics and vision code.

### 1. Keep algorithmic steps small and swappable

The vision stack is naturally a sequence of distinct steps:

- acquire data
- detect candidates
- estimate geometry and kinematic state (i.e. velocity, position, angular velocity)
- select a target
- communicate a result to MCU or devboard

Instead of putting all of that into one giant loop, AutoX breaks those steps into modules. For example:

- `ClassicalDetectorModule` is responsible for producing `panels`
- `SelectingWith3DModule` consumes `panels` and produces `target_panel`

That makes it easy to:

- replace one algorithm without rewriting the whole engine
- compare alternate implementations
- reuse one step in multiple engines
- reason about failures at a smaller scope

### 2. Make data dependencies explicit

Modules declare their dependencies as strings in `inputs` and `outputs`. That may feel simple, but it gives the framework a clear, inspectable contract.

For example, the selection module says it needs `panels` and will output `target_panel`. The detector says it outputs `panels`. That makes the intended flow obvious even before reading the algorithm details.

The shared context type, such as `AutoAimContext`, acts like the schema for the pipeline. It tells you which fields are expected to exist in that engine's world.

### 3. Support mock and real behavior with one public API

One of the design goals of the repo is to run mock behavior by changing configuration instead of rewriting call sites.

The module decorators support that directly:

- `@real(...)` marks the hardware-backed or production implementation
- `@mock` marks the simulated implementation

The selection happens inside the base class, based on `config.mock.enable` from `info.yaml`. Because engines only call `module.run()`, the caller does not change when switching between real and mock behavior.

This is especially useful for:

- laptop development without hardware
- demos that need deterministic behavior
- in the future a way to determine modules based on the hardware profile of the device

### 4. Keep most logic pure-ish even when the system is not

Robotics systems always have side effects: cameras, IPC, serial, display windows, device drivers, and timing constraints. Modules push the codebase toward a cleaner shape by making each step look like:

`context in -> compute -> context out`

Even when a module touches external systems, it still exposes a small contract to the rest of the framework.

### 5. Allow reuse through remapping

The base class also provides `remap_inputs()` and `remap_outputs()`. That is a hint about the intended design: a module should be reusable in more than one pipeline or context layout without rewriting the implementation itself.

## How a module works

Every module subclasses `Module[T]`, where `T` is a `Context` type.

Typical structure:

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

At runtime:

1. `run()` is called
2. the base class selects the registered `@real` or `@mock` method
3. declared inputs are pulled from the context by name
4. the decorated method is executed
5. returned values are written back to the declared output fields

This means most module authors only need to care about the algorithm itself.

## Real example: `ClassicalDetectorModule`

`ClassicalDetectorModule` is a good example of what this abstraction buys us.

It declares:

- `inputs=[]`
- `outputs=["panels"]`

That tells us it is an entry-point style module: it does not depend on earlier module outputs, and it populates the context with detections.

Its `@real(requires="camera")` implementation:

- grabs a frame from the video stream
- runs classical image processing
- pairs light bars into armor candidates
- estimates geometry
- returns a list of `ArmorPanel` objects

That is a full vision stage, but the outside world only sees one simple contract: after running, `ctx.panels` is updated.

## Real example: `SelectingWith3DModule`

`SelectingWith3DModule` shows the next stage in the chain.

It declares:

- `inputs=["panels"]`
- `outputs=["target_panel"]`

So its role is not detection. Its role is decision-making.

Given a list of detected panels, it scores candidates using:

- screen position
- apparent size
- distance from the previous target
- estimated depth

The rest of the engine does not need to know that scoring logic. It only knows that the module turns `panels` into `target_panel`.

## Why modules use a shared context instead of passing many arguments

The shared context solves a practical robotics problem: different stages often need access to overlapping state, and that state grows over time.

Using a typed context makes it easier to:

- add new intermediate values without changing every method signature
- inspect the entire state passed through an engine
- preserve previous-step results such as `prev_target_panel`
- keep the pipeline shape readable

`AutoAimContext` is a good example. It holds the evolving world state for the auto-aim engine, including detected panels and the currently selected target.

## Design tradeoffs

This abstraction is intentionally lightweight, but it does have tradeoffs.

Benefits:

- modular algorithm design
- easier mocking
- clearer data flow
- better reuse
- easier testing at the step level

Costs:

- field names are string-based, so typos matter
- context updates are dynamic
- modules still need discipline to stay focused and not grow too large

Even with those tradeoffs, the abstraction makes sense for this repo because the project needs to switch between real and simulated behavior, and because the processing stack is naturally composed of small stages.

## In short

Modules exist so AutoX can treat each algorithmic step as a replaceable building block with a stable interface.

That gives the project:

- cleaner vision and control pipelines
- easier mock-vs-real switching
- reusable components across engines
- a shared data model that can evolve as the robot software grows


