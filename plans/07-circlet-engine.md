# 07 — Circlet: a 4-camera ring feeding 360° awareness to auto-aim

## Goal
Add a second engine, **circlet**, that runs a ring of body-fixed cameras through
the classical detector and feeds their detections to `FullStateAutoAimEngine` so
its `classification` + `targeting` can decide over a 360° view (react to robots
the gimbal camera can't see). Plus modernize capture (PyAV + v4l2) and add a
looping video mock so the whole thing is testable on a laptop.

## Status (DONE — feature/circlet)
Landed in three commits:

**Phase 1 — capture backends + mock.** `src/drivers/video_sources.py`:
`PyAvCameraSource` (in-process libav decode + `v4l2-python3` controls, the
optional `linux` extra, lazy-imported), `MockVideoSource` (loops a configurable
video, resized to config, runtime-variable fps, synthetic fallback for un-pulled
git-lfs clips), ffmpeg `CameraSource` retained. `CameraDriver`/`FrameReader` are
now source- and camera-agnostic (`resolve_camera_config(name)` threads
index/backend/width/height/fps through `provision`→`from_conn`→`client`). Config
`hardware.video_backend`/`mock_*`, `@MOCK_CAM`. Doc: `docs/camera-driver-testing.md`.

**Phase 2 — sim test for classification/targeting.** `tests/sim/scene.py`
(position-level scene mock: ground-truth robots, fake cameras at async/variable
fps) + `tests/test_classification_targeting_sim.py` (multi-robot bucketing,
closest-target, async invariance, moving target switch).

**Phase 3 — cross-engine pub/sub + the engine.** See below.

## Decisions (locked)
- **Circlet pre-transforms to the CHASSIS frame** (each camera's static extrinsic
  only — no `McuClient` in circlet). Auto-aim lifts chassis→turret with the live
  gimbal yaw/pitch it already gets from the MCU.
- **Transport = pickled `multiprocessing.Queue`** (the plan-03 interim shortcut):
  engines declare `publishes_queue` / `subscribes_queue`; `launch_system` makes
  one shared queue per name; two publishers of one name = hard startup error.
  Last-value semantics via `Engine.publish` / `Engine.latest_subscribed`.
- **Division of labor:** circlet broadens target *selection*; the gimbal-camera
  KF + ballistic solve on the selected target is unchanged. Circlet detections do
  **not** feed the learning KFs.
- **Cameras are unsynchronized** — each handled independently per loop. Ring = 4.

## Phase 3 components
- `src/types/circlet.py` — `CircletPanel` / `CircletDetections` (picklable;
  contour/bbx dropped).
- `src/subsystems/circlet_support.py` — detector-free (mrcal-free) geometry +
  classification: `extrinsic_from_cfg`, `panel_to_chassis`,
  `chassis_to_turret_matrix`, `circlet_panels_to_robots`. Unit-tested.
- `src/engines/circlet.py` — `CircletEngine`: 4 `CameraDriver`s, one detector +
  context per camera, transform to chassis, publish `CircletDetections`.
- `src/core/engine.py` + `orchestrator.py` — the queue-link mechanism.
- `src/engines/full_state_autoaim.py` — `_consume_circlet()` drains the link,
  lifts chassis→turret with the cached gimbal pose, classifies by icon
  (`_circlet_robots` / `_circlet_target`). Opt-in `_slew_to_circlet()`
  (`circlet.drive_slew`, default **off**) aims at a circlet-only target when the
  gimbal camera has none (aim, hold fire).
- `src/info.yaml` — `circlet:` block (cameras, fps, loop_hz, drive_slew) +
  `@CIRCLET`.

## Deviations from the original plan (and why)
The original plan had auto-aim **merge circlet panels into `ctx.panels`** before
classification. Implemented instead as a **separate, additive selection layer**
(`_consume_circlet` builds circlet robots in parallel; the gimbal `ctx.panels`
flow is untouched) because: (a) mixing chassis-frame circlet panels with
camera-frame gimbal panels in one list would be double-transformed by the
existing MCU step, and (b) it keeps the proven gimbal aim path risk-free while
this lands without on-robot validation. Actuation (`drive_slew`) is therefore
opt-in and off by default.

## NOT done — explicit future work (validate on hardware)
- **Per-camera extrinsic calibration.** Extrinsics are config stubs
  (`yaw_deg`/`translation`, identity default). Calibrate camera→chassis per ring
  camera.
- **`chassis_to_turret_matrix` signs/axis order** are a first approximation —
  validate against the real gimbal convention before enabling `drive_slew`.
- **Per-camera intrinsics.** `pnp.py`/`camera_info` use one global intrinsics;
  the ring shares it for now.
- **iceoryx2 `IpcType` zero-copy transport** (plan-03 proper) if the queue ever
  becomes a bottleneck — today's rates (~4×15 fps) don't need it.

## Verification
- Unit (no camera/mrcal): `uv run pytest tests/test_mock_video_source.py
  tests/test_classification_targeting_sim.py tests/test_circlet_engine.py`.
- On the Jetson/laptop (mrcal): `uv run main.py @CIRCLET @MOCK_CAM @SENTRY` — logs
  show circlet publishing panels/s and auto-aim consuming them; `git lfs pull`
  for real footage (else synthetic frames). See `docs/camera-driver-testing.md`.
