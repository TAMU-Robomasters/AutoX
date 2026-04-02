# Engines

## What an engine is

An engine is the process-level unit of work in AutoX.

If modules are the individual algorithmic building blocks, an engine is the thing that owns those blocks and runs them as a live system.

In practice, an engine is responsible for:

- owning an ordered set of modules
- owning the context passed through those modules
- managing long-running loop behavior
- handling process lifecycle
- creating process-local resources such as IPC publishers, subscribers, and device handles

The base class in `src/core/engine.py` inherits from `multiprocessing.Process`, so an engine is meant to run as its own OS process.

## Why this abstraction exists

This layer exists because robotics code has two very different concerns:

1. **algorithmic steps** such as detection and target selection
2. **runtime system behavior** such as process startup, IPC

Modules solve the first problem. Engines solve the second.

Without an engine abstraction, each feature would need to reimplement its own process loop, startup code, shutdown behavior, module wiring, and resource management. That leads to duplicated and inconsistent process code.

By separating the two layers, AutoX gets a cleaner architecture:

- modules describe **what computation happens**
- engines describe **how and where that computation runs**

## Why engines are processes

AutoX is built for systems where different subsystems may need to run independently and communicate through IPC. The engine abstraction makes that explicit by inheriting from `multiprocessing.Process`.

You can see this in `src/main.py`, where separate engines are created and started independently.

## Why `initialize()` is separate from `__init__()`

This is one of the most important reasons the engine layer exists.

The base class intentionally asks subclasses to put process-local setup in `initialize()` instead of `__init__()`. The reason is multiprocessing safety.

In robotics code, `__init__()` is a dangerous place to open resources such as:

- cameras
- serial ports
- IPC nodes
- GUI/display windows

If those are created in the parent process and then copied into a child process, the behavior can become fragile or outright broken. By delaying real setup until `run()` executes inside the child process, the engine creates its resources in the correct place.

That is why the base class structure is:

1. construct engine object
2. `start()` the process
3. child process calls `initialize()`
4. child process repeatedly calls `execute()`

This is a strong signal that the engine abstraction was made to safely support real hardware and IPC in a multiprocessing system.

## What the engine base class provides

The base `Engine` class provides a few framework-level guarantees.

### 1. A standard lifecycle

Every engine follows the same lifecycle:

- `initialize()` for setup
- `execute()` for one iteration of work
- `run()` for the main loop
- `stop()` for shutdown/join behavior

That makes different engines feel uniform even when their internal logic is very different.

### 2. Wiring validation

Before the engine runs, it validates the module/context contract.

It checks that:

- every declared module input/output exists on the context type
- every required input is produced somewhere in the module set

This reduces a common class of bugs in pipeline-style systems: silent mismatches between expected and produced data.

### 3. A stable interface for the orchestrator

Even though the current `orchestrator` file is still mostly a placeholder, the engine base class clearly defines the contract an orchestrator would need:

- start an engine
- stop an engine
- treat it as a named, manageable runtime unit

That suggests the abstraction was designed with future multi-engine orchestration in mind.

## How engines relate to modules

An engine groups modules around one mission-level goal.

Examples from this repo:

- an auto-aim engine groups detection and target-selection modules
- a counting engine uses the same engine lifecycle even though it has no modules

This shows the intended role of an engine: it is not just a pipeline container. It is the full runtime shell around a feature.

The engine decides things like:

- which modules are used
- in what order they run
- when they run
- what other side effects happen around them

That is why the auto-aim engine calls module `run()` methods directly inside `execute()` rather than only delegating to `Pipeline`. The engine needs room for surrounding logic like IPC publishing and display updates.

## Real example: `SimpleAutoAimEngine`

`SimpleAutoAimEngine` shows the design clearly.

It owns:

- an `AutoAimContext`
- a `ClassicalDetectorModule`
- a `SelectingWith3DModule`

At construction time, it declares the module set and context type.

At runtime:

- `initialize()` creates the IPC node and log publisher
- `execute()` runs detection
- `execute()` runs target selection
- `execute()` publishes a log message
- `execute()` updates display windows

That mix of algorithmic work and runtime side effects is exactly why the engine layer exists. A module alone should not be responsible for owning the whole process loop.

## Real example: `AdvancedAutoAimEngine`

`AdvancedAutoAimEngine` is a useful clue about the intended flexibility of the abstraction.

It reuses the same engine shell but swaps the module composition down to only detection right now. That suggests engines are meant to be easy to specialize by changing the module set while keeping the same process semantics.

In other words, AutoX is aiming for:

- one common runtime model
- many interchangeable engine implementations

## Real example: `CountEngine`

`CountEngine` is especially helpful for understanding intent because it has **no modules at all**.

It still fits the engine abstraction because it still needs:

- process startup
- IPC setup
- repeated work inside `execute()`
- graceful stop behavior

That tells us the engine layer is not only about module pipelines. It is about representing a long-running subsystem as a managed process.

## Why not just use a plain while-loop script?

For a small prototype, a plain loop is enough. For a robotics system, it breaks down quickly.

The engine abstraction adds structure around:

- concurrency
- ownership of hardware or IPC endpoints
- clean startup boundaries
- reusable runtime patterns
- future orchestration of multiple subsystems

That makes the codebase easier to scale from one script into a collection of cooperating runtime services.

## In short

Engines exist so AutoX can package a feature as a managed, multiprocessing-friendly runtime unit.

That gives the project:

- a consistent process lifecycle
- safer hardware and IPC initialization
- a clean home for groups of modules
- validation around module/context contracts
- a path toward orchestrating multiple robot subsystems together

