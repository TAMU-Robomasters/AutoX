# 03 — Cross-engine pub/sub + serialization (IpcType)

## Goal
Let engines exchange context fields across process boundaries the same way
modules exchange them within an engine — declaratively, by name — without modules
knowing IPC exists. Built on iceoryx2 (fixed-size), with serialization living on
the types themselves.

## Why
Today wiring is per-engine only (`Engine._validate_wiring`). To have, say, the
AutoNav brain consume AutoAim's `estimate`/`engage_intent`, or a viz engine consume
everyone's context, engines need a publish/subscribe layer at the engine boundary.

## Decisions (locked)
- **Everything on iceoryx2 is fixed-size ctypes.** Subscriber reads are read-only.
- **Serialization is methods on the types** in `src/types`, via an ABC:
  - `IpcType` (ABC) with `to_iceoryx2()` / `from_iceoryx2(payload)` / `to_rerun()`,
    defaulting to warn+return-None; each concrete type overrides what it can.
  - ABC chosen over `getattr`-by-string dispatch — faster (~110 ns vs ~254 ns/call,
    benchmarked) AND visible to `ty`/mypy.
- Cross-engine links are **explicit** (engines declare what they publish/subscribe),
  and **two engines publishing the same field name is a hard startup error**
  (the safety net for the global-namespace problem, since the topology CLI is dropped).

## Design
- `src/types/ipc_type.py` — `IpcType` ABC. Each context field type that crosses
  engines (e.g. `RobotStateEstimate`, a `Waypoint`, an `EngageDirective`) subclasses
  it and implements a fixed-capacity ctypes payload + `to/from_iceoryx2`. Pattern to
  follow: the existing fixed arrays in `EmbeddedTransformationMessage` (16-float
  matrix) and the `FramePayload` pixel array. Variable-length things (e.g. `panels`)
  get a `MAX_N` fixed-capacity array + a `count` field.
- Engine declares `publishes: list[str]` / `subscribes: list[str]` (subsets of its
  module outputs / inputs respectively).
- `src/core/engine_io.py` — `EnginePublisher`/`EngineSubscriber` wrapping one
  iceoryx2 publish-subscribe service per published field (deterministic name,
  e.g. `autox/field/<name>`), last-value semantics, created in the child.
- `Engine.run()` loop becomes: `subscriber.apply(ctx)` -> `execute()` -> `update()`
  -> `publisher.publish(ctx)`. Modules inside `execute()` are unchanged and never
  see IPC. (By the time modules run, subscribed inputs are already on `ctx`.)
- **Redesign `_validate_wiring`**: an input is satisfied by a local module output OR
  a `subscribes` entry; still raise on genuine typos; every `publishes`/`subscribes`
  name must be a context field and actually produced/consumed locally; duplicate
  publisher of a field across engines = hard error.

## Build order
1. `IpcType` ABC + codecs for the first field that actually crosses engines
   (`estimate` and the engage directive from plan 02).
2. `engine_io.py` + the `Engine.run()` apply/publish steps + `publishes/subscribes`
   declarations + `_validate_wiring` redesign.
3. Split AutoAim's `estimate`/`engage_intent` to the AutoNav brain as the first
   real cross-engine link.

## Verification
- Two engines exchange a field over iceoryx2 with last-value semantics on the
  Jetson; per-engine validation still passes; duplicate-publisher raises; orphan
  subscribe raises; round-trip a codec in a unit test.

## Interim shortcut
For low-rate rich links that are awkward to fix-size early, a pickled
`multiprocessing.Queue` between two engines is acceptable (mirrors the old
float-Queue). Reserve iceoryx2 + `IpcType` for the high-rate/latency-sensitive
links. Make it visible which links are zero-copy vs queue.
