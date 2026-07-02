# AutoX refactor plans

Working notes for the camera/driver/IPC refactor, written so a new session can
pick up without re-deriving everything. Read this file first, then the numbered
plan for whatever you're tackling.

## Where things stand (DONE)

**Camera as a shared driver, zero-copy, ffmpeg capture**
- `src/drivers/video_stream.py`:
  - `CameraSource` — in-process capture via an **ffmpeg subprocess** (MJPEG ->
    raw BGR over a pipe) + a daemon reader thread. ffmpeg multi-threads the JPEG
    decode and hits the camera's full **~90 fps**; OpenCV's built-in decode
    capped at ~72. Manual exposure set via `v4l2-ctl`. ffmpeg is killed-with-parent
    (`PR_SET_PDEATHSIG`) so a terminated driver never orphans it on the camera.
  - `CameraDriver(Driver)` — owns a `CameraSource`, copies the newest frame into
    a loaned **iceoryx2** shared-memory slot, publishes into a fixed-size ring
    buffer (`enable_safe_overflow`, `history_size`, `subscriber_max_buffer_size`).
  - `FrameReader` — subscriber; `latest()` drains to the newest frame and returns
    a **read-only, zero-copy** `Frame` view (`writeable=False`; `.copy()` to draw).
  - `make_frame_payload_type(w,h)` — fixed-size ctypes payload, sized from config.
  - `camera_info` — static accessor: `.width/.height/.intrinsics()` from config +
    `opencv12.cameramodel` (lazy mrcal). **No camera, no IPC.** Use it anywhere.
  - iceoryx2 is imported lazily inside the driver/reader, never at module import.
- iceoryx2 pinned `==0.9.0` in `pyproject.toml` (0.9.1 has no sdist; builds from
  source on aarch64 via uv+puccinialin, ~3 min, no sudo).

**Engine/driver framework**
- `Engine` declares `drivers = {name: DriverType}`; `__init__(..., driver_registry)`;
  `run()` calls `_build_driver_handles()` (builds client handles in the child)
  before `initialize()`, then calls every `module.initialize()`; `engine.driver(name)`
  returns the handle.
- `Driver` contract: `provision(name)->conn` (parent), `from_conn(conn)->driver`,
  `client(conn)->handle` (child). `CameraDriver` implements all three.
- `orchestrator.launch_system([EngineClasses])` — spawns one shared driver per
  declared name, wires the registry, starts engines, returns all processes.
- `Module.initialize()` — one-time per-module child-process setup hook (no-op
  default); replaces "lazily start on first run()".
- `Driver.stop()` fixed (uses a `multiprocessing.Event` + join-timeout/terminate).

**Cut over to the new camera path**
- `detection_test_engine` and `particle_filter_autoaim` both: declare
  `drivers={"frames": CameraDriver}`, read `FrameReader.latest()`, write
  `ctx.frame`/`ctx.frame_ts`; the detector reads `ctx.frame`. PF is frame-paced
  (waits for a new `seq`) so it processes each frame once and doesn't flood the MCU.
- `pnp.py` uses `camera_info` (dead `unproject`/`warp` branches removed).
- `display.py` uses `camera_info`; `Window.set_image()` copies (so read-only shm
  frames are safe to annotate); detector copies only when `display_live_frames`.
- `main.py` and `orchestrator.start_engines()` launch via `launch_system`.
- Old `src/subsystems/video_streaming/` (mrcal lens modes, realsense/webcam/
  simulation, depth) snapshotted to `archive/video_streaming/`.

## Locked decisions (don't re-litigate)

- **Always shared** for frames (iceoryx2). No in-process-vs-IPC auto-selection.
- **Drivers live at the engine level; modules only touch `ctx`.** No module-owned
  hardware (kill the `CppDetectorModule._ensure_started` pattern as you migrate).
- **Everything on iceoryx2 is fixed-size ctypes.** Subscriber reads are read-only.
- **Serialization lives in `src/types` as methods on the types** via an ABC
  `IpcType` with `to_iceoryx2`/`from_iceoryx2`/`to_rerun` (default warn+None,
  override per type). ABC chosen over getattr dispatch (faster + type-checkable;
  benchmarked ~110 ns vs ~254 ns/call).
- **Camera: normal lens mode only.** Lens correction/intrinsics reading kept
  minimal (`camera_info`); the rest is archived.
- **MCU stays engine-level** (direct calls, like PF does with `EmbeddedCommunicator`)
  until the `McuDriver` lands (plan 01).
- **ROS is optional, behind a profile, rclpy imported lazily.** ROS<->AutoX is
  lightweight (no shared memory); shared memory is only for camera frames + viz.
- **Behavior brain is a `py_trees` tree folded into the AutoNav engine**, kept as a
  pure rclpy-free object. Engage is **fail-OPEN** (brain/nav2 dead -> keep shooting);
  the only hard local interlock is **match-active** (pre-match OR post-match-end)
  read from the devboard and enforced in AutoAim. Waypoints via `NavigateToPose`
  action; "wiggle at waypoint" via small relative goals, never `/cmd_vel`.

## Outstanding (the numbered plans)

1. `01-mcu-driver.md` — MCU bridge: queue-RPC `McuDriver` + `McuClient`, SERIAL/
   MOCK backends, profile `MCU=`, migrate PF off the inline `EmbeddedCommunicator`.
2. `02-ros-autonav-behavior.md` — `RosBridge` (lazy rclpy), `AutoNav` engine wrapping
   nav2 + the py_trees brain, engage coordination, nav2 watchdog/revive.
3. `03-cross-engine-pubsub.md` — cross-engine publish/subscribe over iceoryx2 +
   the `IpcType` serialization ABC + `_validate_wiring` redesign.
4. `04-visualization-rerun.md` — rerun viz engine + opt-in per-loop context publish.
5. `05-cleanup-and-fixes.md` — archive the old `video_streaming` layer, `Engine.stop()`
   bug, info.yaml cleanup (dead `zed`, unify width/height names), restore PF plot.
6. `06-logging-migration.md` — finish moving prints onto the new multiprocess
   logging (`src/toolbox/logger.py`): sweep remaining files, wire the log queue
   into driver processes, retire `config.log.disable_print`.

## Recurring gotchas

- AutoX's own autobooted **`autox_boot.service`** runs `uv run main.py` on boot and owns
  the camera + iceoryx2 publisher; a second manual `uv run main.py` collides
  (`ExceedsMaxSupportedPublishers`, blank display). `main.py` now guards against it
  (`src/toolbox/autoboot_check.py`) — free the camera for dev with
  `sudo systemctl stop autox_boot.service` (returns on reboot; `./utils/kill_onboot_cv` to
  disable for good). Installed by `utils/setup_boot_script.sh`.
- The autobooted **`cv_dark_boot.service`** (Armor-Panel-Classical, a *different* repo)
  grabs the camera on boot. Free it with `sudo systemctl stop cv_dark_boot.service` (sudo
  password in `~/.pass`). It returns on reboot.
- Don't `pkill -f <string>` where `<string>` appears in your own command line — it
  self-matches and kills your shell. Kill by PID.
- Engines that declare `drivers` MUST be launched via `launch_system`, not
  constructed and `.start()`ed directly (you'll get a clear RuntimeError now).
