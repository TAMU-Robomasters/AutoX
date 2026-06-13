# 06 — Logging migration (finish moving off `print`)

## Where things stand (DONE, on `feature/radii-estimation`)

The multiprocess logging core is implemented in `src/toolbox/logger.py`,
following the Python 3.10 logging-cookbook "logging to a single file from
multiple processes" pattern:

- `start_log_listener()` — parent-side (called by `orchestrator.launch_system`):
  one `multiprocessing.Queue` + a `QueueListener` thread that owns the real
  handlers (console always; `RotatingFileHandler` when `log.to_file`, path from
  `(path_to) log_file`). Idempotent; `stop_log_listener()` flushes in
  `start_engines()`'s `finally`.
- `configure_child_logging(queue)` — child-side, called first thing in
  `Engine.run()`. Attaches a single `QueueHandler` to the `autox` root logger;
  level from `config.log.level`; per-logger overrides from `config.log.levels`
  (e.g. `{panel_tracking: DEBUG}` to debug one module without drowning in
  another's output); `config.log.disable_all_logging` -> `logging.disable()`
  (near-free integer check per call — the perf kill switch).
  `queue=None` (tests/scripts) falls back to a console handler.
- `get_logger(name)` -> `autox.<name>`; every `Module` has `self.log`
  (`autox.<module name>`), every `Engine` has `self.log`. **Gotcha handled:**
  ROS 2's `launch.logging` hijacks `logging.setLoggerClass` with loggers that
  default `propagate=False`; `get_logger` forces `propagate=True` or records
  silently vanish whenever ROS is sourced (MCU=ROS, launch_testing in pytest).
- The log queue rides an attribute (`engine._log_queue`) set by `launch_system`
  after construction, inherited by the child at fork — engine subclass
  constructors didn't need a signature change.
- Adopted by: the new `FullStateAutoAimEngine` (logs every state transition at
  INFO — the main debugging artifact for constant learning), all new/rewritten
  modules, `JsonStore`, `robot_constants`. `Engine` also logs the
  module->inputs->outputs pipeline table at INFO on startup.
- Hot-loop rules (documented in `logger.py`): lazy `%` args, never f-strings in
  per-frame paths; `log.isEnabledFor(DEBUG)` around expensive formatting.
- Tests: `tests/test_logger.py`.

## Remaining work

1. **Sweep the rest of the `print`s.** Mechanical migration, file by file:
   detection/vision modules, drivers (`video_stream.py`, `mcu.py`),
   classification/targeting, `pf.py`, the archived engine (optional), tools in
   `utils/`. Rules of thumb:
   - per-frame chatter -> `self.log.debug(...)` (lazy args!)
   - state changes / acquisitions / mode switches -> `info`
   - degraded-but-running (no transform, solver fail) -> `warning`
   - `print` for actual stdout output (CLI tools) stays `print`.

2. **Driver processes.** Drivers (`CameraDriver`, `McuDriver`) are separate
   processes and currently still print. Extend the driver factory contract so
   `launch_system` can hand them the queue, mirroring engines:
   - simplest: set `driver._log_queue = queue` after `from_conn()` (same
     attribute-inheritance trick as engines) and call
     `configure_child_logging(self._log_queue)` at the top of `Driver.run()`.
   - then migrate driver prints (`[camera]`/`[mcu]` prefixes become logger
     names: `autox.camera_driver`, `autox.mcu_driver`).

3. **Retire `config.log.disable_print`** once no per-frame prints remain in the
   hot path; the global print-swap hack in `toolbox/globals.py` can then go.

4. **Optional: frame-correlated records.** A `logging.Filter` that stamps the
   current frame `seq` (engine sets it on a contextvar each loop) into the
   format string would let you line up records across engines per frame —
   pairs nicely with the rerun viz plan (04-visualization-rerun.md).

5. **Optional: per-engine files.** If one rotating file gets noisy with
   multiple engines, the listener can fan out per-`processName` files — keep
   the single queue, add a routing handler in `_build_handlers()`.
