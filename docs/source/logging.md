# The logging system

AutoX runs each engine and driver as its own OS process, which makes logging
trickier than a single-process script: separate processes can't safely share a
file or console handler. The logging system (`src/toolbox/logger.py`) solves this
with the standard library's queue-based pattern, so all processes' logs end up
interleaved correctly in one place.

## The model: one queue, parent owns the handlers

It follows the Python logging cookbook recipe "logging to a single file from
multiple processes":

```
child process A ─┐
child process B ─┼─► shared mp.Queue ─► QueueListener (parent thread) ─► console
child process C ─┘                                                    └─► rotating file (optional)
```

- The **parent** (the orchestrator) calls `start_log_listener()` once. It creates
  the shared `multiprocessing.Queue` and a `QueueListener` thread that owns the
  *real* handlers — the console handler and an optional rotating file handler.
- Each **child** process calls `configure_child_logging(queue)`, which attaches a
  `QueueHandler` to the `autox` root logger. The child does no formatting or I/O;
  it just enqueues log records, and the parent's listener formats and writes them.
- `Engine.run()` calls `configure_child_logging` automatically at the top, so
  **every engine's logs are wired up for free** — you don't set this up yourself.

This is why you should launch engines through `launch_system` (which calls
`start_log_listener` and threads the queue into each engine): it's what makes the
multiprocess logging actually work.

## Getting a logger

You rarely create a logger by hand:

- **Modules** already have `self.log` (named `autox.<module name>`), made in
  `Module.__init__`.
- **Engines** have `self.log` (named after the engine class).
- Anywhere else, call `get_logger("something")` — it returns a logger namespaced
  under `autox.`.

```python
self.log.info("anchored %r -> FULL_STATE_TRACKING", name)
self.log.debug("estimate: %s", self.ctx.estimate)
```

Everything hangs off the single `autox` root logger, so one set of handlers (and
one config block) controls all of it.

```{note}
`get_logger` forces `propagate = True`. That's deliberate: ROS 2's
`launch.logging` hijacks the global logger class and defaults loggers to
`propagate=False`, which would silently swallow our records whenever ROS is
sourced (`MCU=ROS`, launch tests under pytest, ...). Forcing propagation keeps
logs flowing in those setups.
```

## Configuration (`info.yaml` `log:` block)

All logging behavior is config, not code. The relevant keys under `(default)`:

```yaml
log:
    disable_all_logging: false  # kill switch: logging.disable() everywhere (perf)
    level: DEBUG                # root level for all autox.* loggers
    levels: {}                  # per-logger overrides, e.g. {panel_tracking: DEBUG}
    to_file: false              # also write to the (path_to) log_file (rotating)
    max_file_mb: 20             # rotating file: size before rollover
    backup_count: 3             # rotating file: how many old files to keep
```

The two most useful in practice:

- **`levels` (per-logger overrides).** Crank up one noisy module without drowning
  in everything else, e.g. `levels: {panel_tracking: DEBUG}` to debug panel
  tracking while the rest stays at `level`. Logger names match the module/engine
  names (`get_logger`'s argument).
- **`to_file`.** Off by default (console only). Turn it on to also write a
  rotating log file at the `(path_to) log_file` location — handy for after-match
  review on the robot, where there's no console to watch.

`disable_all_logging` is a hard kill switch via `logging.disable`: when a level is
disabled, the call is a near-free integer check, so this is the option to reach
for if logging ever shows up in profiling.

```{note}
The same `log:` block also holds a few non-logging frame/debug toggles that share
the namespace historically — e.g. `display_live_frames`, `save_frame_to_file`,
`save_rate`. Those drive the display/recording paths, not the stdlib logger.
Profiles override them per board: `BOARD=LAPTOP` sets `save_frame_to_file: true`
and `save_rate: 1`, for instance.
```

## Hot-loop discipline

The auto-aim loop runs tens of times per second, so logging hygiene matters:

- **Use lazy `%`-style args, never f-strings, in per-frame paths.**
  `log.debug("x=%s", x)` defers the string formatting until the record is actually
  emitted; `log.debug(f"x={x}")` pays the formatting cost every call even when
  `DEBUG` is disabled.
- **Guard expensive formatting** with `if log.isEnabledFor(logging.DEBUG):` before
  building a big debug string.

You can see both rules and the two-tally FPS logging in
`full_state_autoaim.py` (`_log_fps`, and the `log.debug(...)` calls that use
`%`-args).
