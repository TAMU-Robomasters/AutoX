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
| `src/engines/circlet.py` | `CircletEngine`. `_CIRCLET_N=4`, `drivers={cam_0..cam_3}`, `publishes_queue`. **Headless.** Per-loop: detect each fresh-seq cam → `panel_to_chassis` → publish. |
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

## Config quick reference (`src/info.yaml`)
- `circlet:` — `enable`, `loop_hz`, `timeout_ms`, `drive_slew`, `width/height/fps`,
  `backend` (`pyav|mock|ffmpeg`), `cameras: [{index, yaw_deg, translation}, ×4]`.
- `hardware:` — `video_backend`, `mock_video_path`, `mock_fps`, `mock_fps_jitter`.
- Profiles: `@CIRCLET` (enable + mock backend), `@MOCK_CAM` (mock capture).
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
- Full pipeline (mrcal): `uv run main.py @CIRCLET @MOCK_CAM @SENTRY` — logs show
  circlet publishing panels/s and auto-aim consuming. See `docs/source/camera-driver.md`.
