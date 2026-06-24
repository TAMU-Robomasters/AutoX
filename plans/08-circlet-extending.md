# 08 — Extending circlet (agent orientation)

Companion to **`07-circlet-engine.md`** (design rationale + decisions). This file
is the *how do I build on it* map: data flow, where things live, the extension
points, and the environment gotchas that will bite you. Read `07` for *why*, this
for *where/how*. The feature is **DONE and landed on `feature/circlet`**.

## What circlet is (one line)
A second engine (`CircletEngine`, its own process) running a **ring of 4
body-fixed cameras** at ~15 fps → classical detector → chassis-frame panels →
published to `FullStateAutoAimEngine`, which lifts them to the turret frame with
the live gimbal pose and folds them into target *selection* (360° awareness). The
gimbal KF + ballistic solve on the *selected* target is unchanged.

## Data flow (and coordinate frames)
```
ring cam i ──CameraDriver(cam_i)──▶ FrameReader.latest()        [camera optical frame]
   └▶ ClassicalDetectorModule (per-cam ctx) ──▶ ctx.panels
        └▶ panel_to_chassis(panel, extrinsic_i)                  [camera→chassis 4x4]
             └▶ CircletDetections(panels[], ts)
                  └── Engine.publish("circlet_detections") ──▶ mp.Queue
                                                                  │
FullStateAutoAimEngine.execute():  latest_subscribed() ◀─────────┘
   └▶ _consume_circlet(): chassis→turret via chassis_to_turret_matrix(live yaw,pitch)
        └▶ classify by icon → _circlet_robots / _circlet_target  [parallel candidates]
             └▶ (opt-in) _slew_to_circlet()  iff config.circlet.drive_slew
```
Frames: **camera optical → chassis (published) → turret (in full-state) →
classify/target.** Units cm; x-right / y-fwd / z-up; angles `atan2(y,x)` from +x
CCW; MCU yaw convention appears only at the ballistic output boundary.

**Key design choice:** circlet detections are an **additive selection layer**,
*not* merged into `ctx.panels` (avoids double-transform + keeps the proven gimbal
aim path untouched). Don't "simplify" by merging — see `07` §Deviations.

## File map (where to add things)
| File | Role |
|---|---|
| `src/engines/circlet.py` | `CircletEngine` + `CircletCaptureEngine` / `CircletCaptureAllEngine`. **Headless.** Forks **one worker PROCESS per camera** (`_camera_worker`): each reads its cam zero-copy, detects, `panel_to_chassis`, and pushes a small `CircletPanel` list back over a result queue; the engine aggregates + publishes. Cameras come from the `_camera_names` class attr (override to add cams). See "worker-process model" below. |
| `run_circlet.py` / `run_circlet_capture.py` / `run_circlet_autoaim.py` | Standalone bring-up launchers (repo root). Detect-only ring / capture-only diagnostic / ring+auto-aim. Run with `@CIRCLET_ONLY`. See "Running on real cameras". |
| `utils/circlet_udev.py` | Generates/installs `/dev/circlet/camN` + `maincam` udev symlinks keyed on **USB port path** (stable across `/dev/videoN` renumbering). Re-run `--install` after any re-wiring. |
| `src/types/circlet.py` | `CircletPanel` / `CircletDetections` — picklable wire format (drops `contour`/`bbx`). |
| `src/subsystems/circlet_support.py` | **mrcal-free** geometry/classification: `extrinsic_from_cfg`, `panel_to_chassis`, `chassis_to_turret_matrix` (⚠ STUB), `circlet_panels_to_robots`. Unit-testable with no camera. |
| `src/drivers/video_sources.py` | Capture backends: `MockVideoSource`, `PyAvCameraSource`, `FrameSource` Protocol, `make_source(...)`. |
| `src/drivers/video_stream.py` | `resolve_camera_config(name)`, `_cfg_get`, camera-agnostic `CameraDriver`/`FrameReader`. |
| `src/core/engine.py` | Pub/sub API: `publishes_queue`/`subscribes_queue` + `publish()` / `latest_subscribed()`. |
| `src/core/orchestrator.py` | `_wire_engine_queues()` (one Queue/link, one-publisher rule); launches circlet when `config.circlet.enable`. |
| `src/engines/full_state_autoaim.py` | `subscribes_queue`, `_consume_circlet`, `_closest_circlet_robot`, `_slew_to_circlet`, gimbal-pose caching. |
| `tests/test_circlet_engine.py`, `tests/test_classification_targeting_sim.py`, `tests/sim/scene.py`, `tests/test_mock_video_source.py` | All runnable with **no camera/mrcal**. |

## Extension points (the four you'll actually use)
1. **New cross-engine message.** Set `publishes_queue = "name"` on the producer
   and `subscribes_queue = "name"` on the consumer; call `self.publish(msg)` /
   `self.latest_subscribed()`. `launch_system` auto-wires one shared Queue per
   name. **One publisher per name** (RuntimeError otherwise). Message must be
   **picklable**; semantics are last-value / newest-wins (drains stale). This is
   the plan-03 *interim* link — upgrade to iceoryx2 `IpcType` only if it bottlenecks.
2. **New capture backend.** Implement the `FrameSource` Protocol in
   `video_sources.py`, add a branch to `make_source()`, add a `backend:` value in
   config. `CameraDriver` stays the sole owner — never open capture in a module.
3. **Change how circlet feeds targeting.** Edit `_consume_circlet` in
   `full_state_autoaim.py` (the parallel selection layer). Actuation is
   `_slew_to_circlet`, **off by default** (`circlet.drive_slew`). Don't enable
   slewing until `chassis_to_turret_matrix` is hardware-validated (see gotchas).
4. **Per-camera config.** Each `circlet.cameras[i]` entry overrides the `circlet:`
   defaults; resolution flows through `resolve_camera_config` / `_cfg_get`. The
   ring is fixed at 4 (`_CIRCLET_N`, `drivers`); list index `i` → driver `cam_i`.

## Worker-process model (how the ring actually runs)
`CircletEngine` does **not** detect in its own loop. It overrides `run()` to fork
**one camera-pinned worker process per camera** (right after logging setup, before
any engine-side iceoryx2 node/reader thread exists, so workers inherit loaded
config but no IPC). Each worker builds its **own** `FrameReader`, so frames never
leave shared memory — only the tiny `CircletPanel` list crosses the result queue.
The engine just drains that queue, aggregates the newest list per camera, and
publishes one `CircletDetections`.

- **Why processes, not an in-engine `multiprocessing.Pool`:** a pool had to pickle
  every 2.7 MB frame across the boundary → ~34 Hz ceiling. Per-camera workers put
  the process split *on* the existing zero-copy frame boundary → true ~N× parallel,
  no frame copies.
- **OpenCV thread cap:** each worker calls `cv2.setNumThreads(n_cpu // ring_size)`.
  Without it every worker's OpenCV grabs all cores (N×n_cpu threads ≫ n_cpu) and
  busy scenes thrash. Keep this if you add workers.
- **Extending the ring / adding cameras:** override the `_camera_names` class attr
  (and `drivers`). `CircletCaptureAllEngine` does exactly this — it appends the
  gimbal cam's `"frames"` driver to open all five. Per-camera fps is logged by name.
- **Two diagnostic subclasses** swap the worker via `_worker_target`:
  `CircletCaptureEngine` (`_capture_worker`, NO detector — logs true capture/USB
  fps, publishes nothing) and `CircletCaptureAllEngine` (capture-only, all 5 cams).
- Loop pacing is the `_drain_results` blocking get (≤ one `loop_hz` period), not a
  sleep. Detection (not capture) is the real fps limiter — see below.

## Running on real cameras (hardware bring-up, AGX Orin)
The mock path (`@CIRCLET @MOCK_CAM`) needs no hardware; this is the **real-camera**
runbook. Deep USB/bandwidth detail + the latest physical wiring live in the agent
memory note `circlet-usb-bringup`; the durable rules:

- **Launchers** (all headless, `Ctrl-C` to stop):
  - `uv run run_circlet.py @CIRCLET_ONLY` — the 4-cam ring with detection; logs
    `circlet: detect fps [cam_0=… …]; published … panels/s`.
  - `uv run run_circlet_capture.py @CIRCLET_ONLY` — capture-only, 4 ring cams (true
    USB/IPC fps, no detection). `… @SENTRY @CIRCLET_ONLY` opens **all 5** (adds the
    gimbal cam — needs a robot profile for `config.hardware` fields). The script is
    adaptive: it picks `CircletCaptureAllEngine` iff the gimbal cam is configured.
  - `uv run run_circlet_autoaim.py @SENTRY @CIRCLET_ONLY` — ring + `FullStateAutoAimEngine`.
- **`@CIRCLET_ONLY`** = circlet enable + `backend: pyav` + a **placeholder** shared
  intrinsics (`mrcal_1280x720`, the gimbal cam's) so the detector's PnP runs —
  circlet panel geometry is NOT trustworthy until each ring cam has its own
  intrinsics. It also points the gimbal `camera_index` at `/dev/circlet/maincam`.
  Capture deps are the `linux` extra: `uv sync --extra linux`.
- **Stable device identity.** All these cams are firmware-identical (`32e4:0234`,
  serial `01.00.00` on every unit) → **USB port path is the only discriminator**;
  a serial-based udev rule can't tell them apart, and `/dev/videoN` numbers
  renumber across replug/reboot (an int `camera_index` silently collides → "Device
  or resource busy", a *different* failure from bandwidth ENOSPC). Use the
  `/dev/circlet/*` symlinks from `utils/circlet_udev.py` (config `device:`/path);
  re-run `--install` after re-wiring.
- **USB bandwidth is the hard ceiling, and it's a WIRING problem.** One Tegra xHCI,
  one 480M HS bus, 4 root ports: two USB-C hubs (`1-1`, `1-2`), onboard Bluetooth
  (`1-3`, unusable), and the Type-A bank (`1-4`, an RTS5420 hub **all four dev-kit
  Type-A ports share**). Each UVC cam reserves its **top** isochronous alt setting
  (≈196 Mbps) regardless of requested fps **or** resolution (no bulk endpoint), so
  per-stream bandwidth can't be shrunk in software. Empirical capacity: the Type-A
  bank carries **2** of these cams, each USB-C hub **1–2**. A **3rd cam on Type-A
  ENOSPCs** ("Not enough bandwidth for altsetting 11"). Working 4-cam recipes: 2 on
  Type-A + 1 per USB-C, or 2 per USB-C hub. ⚠ `uvcvideo quirks=128` (FIX_BANDWIDTH)
  is a **proven no-op** for these MJPEG cams — don't reach for it. A **5th camera**
  needs a USB host beyond the built-in ports (PCIe/M.2 USB card with its own
  periodic pool) or MIPI-CSI; that's the gating limit for ring+gimbal on one Jetson.
- **fps is detection-bound, not USB-bound.** Raw capture is ~56 fps on all ports at
  once. With detection on, the per-camera fps spread is **scene-dependent** (the
  classical detector does more contour work on busy/bright views — proven by a port
  swap: slowness travels with the camera/scene, not the port). Lever: `nvpmodel -m 0`
  (MAXN — persists across reboot; `jetson_clocks` does not) and the per-worker
  OpenCV thread cap above; longer term, GPU/lighter detector.

## Config quick reference (`src/info.yaml`)
- `circlet:` — `enable`, `loop_hz`, `timeout_ms`, `drive_slew`, `width/height/fps`,
  `backend` (`pyav|mock|ffmpeg`), `cameras: [{index|device, yaw_deg, translation}, ×4]`
  (`device:` = a stable `/dev/circlet/camN` symlink, wins over `index:`).
- `hardware:` — `video_backend`, `mock_video_path`, `mock_fps`, `mock_fps_jitter`;
  `camera_index` now accepts a **path string** (point the gimbal cam at
  `/dev/circlet/maincam`, not an int — int collides on `/dev/videoN` renumber).
- Profiles: `@CIRCLET` (enable + mock, runs *with* auto-aim), `@CIRCLET_ONLY`
  (enable + pyav + placeholder intrinsics + gimbal symlink — standalone real-camera
  bring-up), `@MOCK_CAM` (mock capture). Stack a robot profile (`@SENTRY`) for the
  gimbal cam's intrinsics/`config.hardware` fields.
- Optional deps: `linux = ["av", "v4l2-python3"]` → `uv sync --extra linux`.

## Dev-environment gotchas (read before you run/extend)
- **mrcal vs Python version.** Repo pins **Python 3.10**; the `python3-mrcal`
  `.deb` installs for the *system* Python (3.12 on Ubuntu 24.04) with
  `cpython-312` `.so`s → **not importable in the 3.10 venv**. Works on the Jetson
  (system Python is 3.10). On a 3.12 laptop, do detection on the Jetson; use
  `@MOCK_CAM` capture + the mrcal-free unit/sim tests locally.
- **Normal-mode runtime is mrcal-free.** `camera_info.intrinsics()` parses
  `opencv12.cameramodel` with a regex (`_read_opencv12_intrinsics` in
  `video_stream.py`) instead of `mrcal.cameramodel(...)`; and `pnp.py` fetches
  intrinsics lazily (first PnP solve). So the whole classical / `normal`-lens-mode
  detection path imports *and runs* with no mrcal. mrcal is only needed for offline
  calibration and the VPI/Jetson `cpp_detector` (top-level `import mrcal`, not on
  the classical path). Don't reintroduce a runtime mrcal import for normal mode.
- **Circlet is headless by design.** It never calls `display.show_windows()`. The
  detector hardcodes the `"main"` window (`module.py`), so all 4 ring detectors
  share one window name — you can't get 4 distinct feeds without per-camera window
  names + a show call in `CircletEngine` (gate behind a config flag if you add it).
  Only `FullStateAutoAimEngine` shows a window (the gimbal cam). So a normal run =
  **1 window, not 5.**
- **Pre-existing test failures (not circlet's):** 5 in `test_aim_state_machine.py`
  from `force_reestimation: true` (`info.yaml:240`, pre-circlet — it makes
  `_state_for` ignore saved constants) + 1 ballistics numeric
  (`test_shot_timing_aims_at_center_with_aim_z`). They became *visible* on
  mrcal-less laptops only because the lazy-mrcal change let those tests collect.
  Don't chase them as regressions.

## Not done — future work (also in `07`)
- ⚠ **`chassis_to_turret_matrix` is a STUB** (`Rx(-pitch)@Rz(-yaw)`, signs/axes
  unvalidated). Validate on the real gimbal before `drive_slew` ever fires.
- Per-camera **extrinsic calibration** (today: identity/`yaw_deg`+`translation`
  stubs) and **intrinsics** (ring shares one global `camera_info`).
- iceoryx2 zero-copy transport (plan-03 proper) if the queue bottlenecks.

## Verification
- Local, no camera/mrcal: `uv run pytest tests/test_mock_video_source.py
  tests/test_classification_targeting_sim.py tests/test_circlet_engine.py`.
- Full pipeline, mock cams: `uv run main.py @CIRCLET @MOCK_CAM @SENTRY` — logs show
  circlet publishing panels/s and auto-aim consuming. See `docs/source/camera-driver.md`.
- Real cameras (Jetson): `uv run run_circlet_capture.py @CIRCLET_ONLY` (capture-only
  sanity, expect ~full-fps on every ring cam) then `uv run run_circlet.py
  @CIRCLET_ONLY` (with detection). If a cam shows fps 0 / ENOSPC it's a wiring/
  bandwidth issue (see the bring-up section), not a code bug.
