# 04 — Visualization engine (rerun)

## Goal
A dedicated visualization engine that receives data from the other engines and
renders it with **rerun** (rerun.io). Programmers annotate frames with OpenCV
markers and ship them to the viz engine; the viz engine decides what to log. Must
be toggle-able from `info.yaml` and add ~zero overhead when off.

## Why / decisions
- Engine base gets an **opt-in (info.yaml) "send my whole context to the viz engine
  every loop"** hook, using each type's `to_rerun()` (the `IpcType` ABC, plan 03).
- Only the camera frame (big) rides **shared memory**; the rest of the context is
  small. ROS<->AutoX and most coordination payloads do NOT use shared memory.
- **Annotated images need a writable copy** — `cv.drawContours`/`putText` mutate in
  place, and shared-memory frames are read-only. `Window.set_image()` already does
  the copy (in `display.py`); the viz path copies once, only when enabled.
- Goal: when viz is disabled, the IPC layer costs nothing on the hot path
  (no copy, no publish).

## rerun specifics (confirmed)
- `rr.log` images (note OpenCV is BGR — tell rerun, or convert).
- Markers via `Points2D` + annotation contexts.
- Custom types via `as_component_batches` / `AnyValues` -> maps directly onto each
  type's `to_rerun()`.

## Design sketch
- `src/engines/visualization.py` `VisualizationEngine(Engine)` — subscribes to the
  per-engine context publishes (plan 03) + a frames `FrameReader`; for each received
  object calls `to_rerun()` and logs to a rerun recording/stream.
- Engine base: a `viz_enabled` flag (from `config.log` / a `viz` config block); when
  set, after `execute()` publish the declared context fields (and optionally an
  annotated frame) to the viz engine. Reuse the plan-03 publish machinery.
- Drawing API: programmers call `display.windows[...].set_image(frame)` then
  `add_contour/add_point/...`; the annotated image is what gets shipped to viz.

## Open questions to resolve when building
- Whether the viz engine pulls (subscribes to everything) or engines push their
  context to a viz service. Pull is cleaner with plan 03's per-field services;
  push is simpler if you just want "dump ctx".
- rerun spawn/connect model on the Jetson (rr.spawn vs serve vs save to .rrd).

## Verification
- `viz` off: assert no extra copy/publish on the detection/PF hot path (fps
  unchanged from ~90).
- `viz` on: frames + markers + a couple of custom types show up in the rerun viewer.

## Depends on
- Plan 03 (`IpcType.to_rerun` + cross-engine publish).
