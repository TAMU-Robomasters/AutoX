# 06 — EKF auto-aim engine for a single, fixed-radius, constant-rate spinning target

This doc is written for someone who has **never touched this codebase before**.
It explains the framework from scratch, walks through a tiny runnable example,
explains the config system, then specs out the actual engine you need to build.
Read it top to bottom the first time.

## 0. The problem, in plain terms (your pipeline doc, mapped to this codebase)

You already wrote up the pipeline at a high level:

1. **Target detection** — camera frame in → position (x, y, z) + orientation of
   each visible panel, relative to the camera. (PnP using camera intrinsics +
   known panel size.)
2. **Transformation** — turret motor angles in → convert that panel pose from
   "relative to camera" to "relative to the turret/robot's global frame."
3. **Tracking** — panel poses in (radius is a *known constant* this time) →
   position, velocity, angular position, and angular velocity of the target's
   center of rotation.
4. **Ballistics** — tracking output + gravity/projectile-speed constants →
   yaw/pitch angles to send to the turret.

Every one of these stages already exists in this repo for a *harder* version of
this problem (unknown radius, unknown/variable spin rate, multiple possible
targets). Your job is **stage 3 only** — replace the existing
"unknown-everything" tracker (a particle filter) with a much smaller
Kalman/EKF-based tracker that exploits the fact that **radius and spin rate are
known constants**. Stages 1, 2, and 4 are reused with little or no change.

The rest of this doc explains *how this repo is organized* (so stages 1/2/4 make
sense to you), then specs out stage 3.

---

## 1. How AutoX code is organized: Engines, Modules, and Context

Before anything else, three words you'll see everywhere:

- **Context** — just a plain Python object (a `@dataclass`) that holds named
  values, e.g. `panels`, `estimate`, `solution`. Think of it as a shared
  clipboard that every step of the pipeline can read from and write to.

- **Module** — one step of the pipeline (e.g. "detect panels," "run the
  tracker," "compute ballistics"). A module says up front: *"I need these named
  values from the clipboard (`inputs`), and I will write these named values back
  (`outputs`)."* It does not know or care what wrote the inputs or what will read
  its outputs.

- **Engine** — the thing that actually runs. It owns one `Context` (the
  clipboard) and a list of `Module`s, and runs them in order, in a loop, forever
  (once per camera frame, typically). An engine runs as its own OS process.

Why split things this way? Two reasons that matter for you:

1. **You can swap one step without touching the others.** The detection,
   transformation, and ballistics modules don't need to change at all when you
   swap the tracker — they only know "something writes `estimate`, something else
   reads `estimate`."
2. **Mock vs. real.** Every module can have a `@real` implementation (uses actual
   camera/GPU/hardware) and a `@mock` implementation (returns fake data). A
   single config flag (`config.mock.enable`) switches *all* modules between them,
   so you could test that nothing crashes on a laptop with no camera/turret attached.

### The rule that will save you debugging time later

**Never open a camera, serial port, or GPU context inside `__init__`.**
`__init__` runs in the *parent* process, before the engine's process is forked.
Anything hardware-related goes in `initialize()`, which the engine calls once
inside the *child* process, right before the main loop starts. (Modules have
their own `initialize()` hook too, called once per module.) If you put camera/GPU
setup in `__init__` it will appear to "sort of work" and then fail in confusing
ways — this is the #1 gotcha in this codebase.

---

## 2. Tutorial: build a tiny module + engine yourself (do this first)

Don't skip this — 15 minutes here will make everything below click. We'll build a
toy engine with two modules: one that counts up, one that doubles the count.

### Step 1 — define a Context

Create `src/types/tutorial.py`:

```python
from dataclasses import dataclass
from typing import Optional
from src.core.module import Context


@dataclass
class CounterContext(Context):
    number: Optional[int] = None
    doubled: Optional[int] = None
```

This is the "clipboard." It has two slots: `number` and `doubled`.

### Step 2 — write two modules

Create `src/subsystems/tutorial_modules.py`:

```python
from src.core.module import Module, real
from src.types.tutorial import CounterContext


class IncrementerModule(Module[CounterContext]):
    """Counts up by one every time it runs. Produces `number`."""

    def __init__(self, context: CounterContext):
        super().__init__(
            name="incrementer",
            context=context,
            inputs=[],            # needs nothing from the context
            outputs=["number"],   # writes ctx.number
        )
        self._count = 0

    @real()
    def _run(self):
        self._count += 1
        return self._count


class DoublerModule(Module[CounterContext]):
    """Reads `number`, writes `doubled`."""

    def __init__(self, context: CounterContext):
        super().__init__(
            name="doubler",
            context=context,
            inputs=["number"],    # reads ctx.number
            outputs=["doubled"],  # writes ctx.doubled
        )

    @real()
    def _run(self, number):
        return number * 2
```

Things to notice:
- `inputs`/`outputs` are just **strings** that must match field names on
  `CounterContext`.
- `@real()` marks "the real implementation." The base class reads
  `self.ctx.<input>` for each name in `inputs`, calls this method with those as
  positional args, and writes the return value(s) to `self.ctx.<output>`.
- Nobody calls `_run` directly — the engine calls `module.run()`, which dispatches
  to `_run` (or to a `@mock`-decorated method, if you wrote one and
  `config.mock.enable: true`).

### Step 3 — write the engine

Create `src/engines/tutorial_counter_engine.py`:

```python
import time

from src.core.engine import Engine
from src.types.tutorial import CounterContext
from src.subsystems.tutorial_modules import IncrementerModule, DoublerModule


class CounterEngine(Engine[CounterContext]):
    def __init__(self, driver_registry=None):
        self.ctx = CounterContext()
        self.incrementer = IncrementerModule(self.ctx)
        self.doubler = DoublerModule(self.ctx)
        super().__init__(
            modules=[self.incrementer, self.doubler],
            context_type=CounterContext,
            driver_registry=driver_registry,
        )

    def initialize(self) -> None:
        pass  # nothing to set up — no camera/serial/GPU here

    def execute(self) -> None:
        self.incrementer.run()
        self.doubler.run()
        print(f"{self.ctx.number} doubled is {self.ctx.doubled}")
        time.sleep(1)


if __name__ == "__main__":
    engine = CounterEngine()
    engine.start()
    try:
        engine.join()
    except KeyboardInterrupt:
        engine.stop()
```

### Step 4 — run it

```sh
uv run python -m src.engines.tutorial_counter_engine
```

You should see `1 doubled is 2`, `2 doubled is 4`, ... once per second, until you
`Ctrl+C`.

### Why it didn't blow up at construction time

When you call `super().__init__(modules=..., context_type=CounterContext, ...)`,
the base `Engine` runs `_validate_wiring()`: it checks that every
input/output name (`number`, `doubled`) is actually a field on
`CounterContext`, and that every input (`number`) is produced as an output by
*some* module (`IncrementerModule` produces it). If you typo a field name, or
forget the producing module, you get a clear `ValueError` here — at startup, not
three steps into a debugging session. This is the main safety net the framework
gives you; keep it in mind when you wire your real modules in §6.

Delete `tutorial.py` / `tutorial_modules.py` / `tutorial_counter_engine.py` once
you're comfortable — they're scaffolding, not part of the real engine.

---

## 3. The config system: `info.yaml` and `local_data.ignore.yaml`

Almost every tunable number in this repo (camera resolution, gravity, projectile
speed, which camera/MCU backend to use, whether to use mocks, ...) lives in
**`src/info.yaml`**, not hardcoded in Python. You'll add your radius/RPM constants
here too.

### Reading config in code

```python
from src.toolbox.globals import config

print(config.ballistic.gravity)              # 981
print(config.ballistic.omega_spin_threshold) # 3.0
```

`config` is just a dict-like object where `.foo.bar` works like `["foo"]["bar"]`.
It's loaded once, at import time, from `src/info.yaml`.

### Structure of `info.yaml`

```yaml
(project):
    (path_to): { ... }              # named filesystem paths
    (local_data): ./local_data.ignore.yaml   # see below
    (profiles):
        (default):
            ballistic:
                gravity: 981
                projectile_velocity: 2300
                omega_spin_threshold: 3.0
            mock:
                enable: False
            ...

        CAMERA=WEBCAM:
            hardware: { camera: webcam, camera_index: 0, ... }

        MCU=MOCK:
            mcu: { backend: mock }
        MCU=SERIAL:
            mcu: { backend: serial }
```

- **`(default)`** is always loaded. It's the base config.
- Every other top-level key under `(profiles)` (`CAMERA=WEBCAM`, `MCU=SERIAL`,
  `GPU=TENSOR_RT`, ...) is an **optional override block**. When you select one on
  the command line, its keys get recursively merged on top of `(default)`.
- You select a profile with an `@` prefix on the command line:

  ```sh
  uv run main.py @CAMERA=WEBCAM @MCU=SERIAL
  ```

  This merges `CAMERA=WEBCAM`'s and `MCU=SERIAL`'s blocks into the config. Forget
  the `@` and `MCU=SERIAL` is parsed as a raw one-off YAML override instead of a
  named profile (works for quick one-off tweaks, but it's easy to typo — prefer
  `@NAME` for anything defined under `(profiles)`).

- **Defaults if you specify nothing**: `GPU=NONE BOARD=LAPTOP CAMERA=NONE
  MCU=MOCK ROS=NONE` (set in `src/toolbox/globals.py`). So if you run plain `uv
  run main.py`, you get `MCU=MOCK` (no serial port opened, identity transform)
  and `CAMERA=NONE` (no camera — you'd add `@CAMERA=WEBCAM` for a laptop webcam).

### `local_data.ignore.yaml`

The first time you run anything, a file `src/local_data.ignore.yaml` is
auto-created (and gitignored — it's *local*, per-machine). It remembers which
`@PROFILE` flags you selected last, under `(selected_profiles)`, so you don't
have to retype `@CAMERA=WEBCAM @MCU=MOCK` every single time. If your config seems
"stuck" on old settings, check this file — it's just YAML, safe to edit or
delete (it'll regenerate).

### `mock.enable`

```yaml
mock:
    enable: False    # True -> every module uses its @mock method instead of @real
```

Flip this to `true` (or pass an override) to run the whole pipeline with canned
data — useful for testing your EKF module's wiring before a camera/turret is
involved. See §11 for what your module's `@mock` should return.

### Adding your own config block

For this engine, add a new top-level key under `(default)` in `src/info.yaml`,
e.g.:

```yaml
spinning_target:
    radius_cm: 23.5   # measure your rig
    omega_rpm: 10     # known constant spin rate
```

then in code: `config.spinning_target.radius_cm`,
`config.spinning_target.omega_rpm`. That's the entire process — no schema, no
registration step.

---

## 4. Reference: walk through the existing pipeline you're adapting

`src/engines/particle_filter_autoaim.py` is the engine you will base the new engine on. Please read alongside it in this
doc. Its `execute()` does, in order:

| # | Step | Module / code | Keep for your engine? |
|---|------|----------------|----|
| 0 | Pull newest frame from the camera | inline, `self.frames.latest()` | **Yes** — identical |
| 1 | Detection: frame → `panels: List[ArmorPanel]` | `ClassicalDetectorModule` | **Yes** (caveat in §10) |
| 2 | Robot classification (sentry/hero/standard) | `RobotClassificationModule` | **No** — drop, single target |
| 3 | Targeting: pick which robot to engage | `TargetingModule` | **No** — drop, single target |
| 4 | Turret-frame transform (camera frame → global/ballistic frame) | inline helper `_transform_panels_to_turret_frame`, lines 247-264 | **Yes** — reuse as-is (§9) |
| 5 | State estimation: panels → `estimate: RobotStateEstimate` | `ParticleFilterEstimationModule` | **Replace** with your `EkfEstimationModule` |
| 6 | Ballistics: `estimate` → `solution: BallisticSolution` | `FullStateShotTimingModule` *or* `FullStateContinuousFireModule`, chosen by `\|ω\| vs omega_spin_threshold` | **Yes**, but always `FullStateContinuousFireModule` — no branch needed (§5) |
| 7 | Send `solution` to MCU, update display/FPS | inline, `self.mcu.send_solution(...)` in `update()` | **Yes** — identical |

`drivers = {"frames": CameraDriver, "mcu": McuDriver}` stays the same — both
engines need the camera and the MCU/turret link. These are **drivers** (§1
mentioned modules; drivers are the analogous abstraction for hardware that only
one process can own — one shared `CameraDriver` process feeds frames to whichever
engines need them). You don't need to understand their internals, just declare
them the same way the PF engine does.

---

## 5. What's different about your target

1. **Orbit radius is known and constant**, and the target is circular, so the
   front/back and left/right panel orbit radii are equal:
   `RobotStateEstimate.a_radius == RobotStateEstimate.b_radius == R`, a fixed
   number you measure once (cm — same units as everything else here). The
   particle filter *estimates* this from particle spread; for you it's a
   constant.

2. **Angular rate is known and constant**: 10 RPM = `10 × 2π / 60 ≈ 1.047 rad/s`.
   This is comfortably below `config.ballistic.omega_spin_threshold` (3.0 rad/s,
   `src/info.yaml:128`), so `FullStateContinuousFireModule` (not
   `FullStateShotTimingModule`) is the correct ballistic module — this matches
   "only wire up continuous-fire."

3. **Single target** — no bucketing by icon (`RobotClassificationModule`), no
   choosing among candidates (`TargetingModule`). Whatever panel(s)
   `ClassicalDetectorModule` returns *are* the target.

These three facts are why a small Kalman/EKF tracker replaces a 40,000-particle
GPU filter.

---

## 6. New pipeline design

### Context

Add a new context dataclass to `src/types/autoaim.py`, e.g. `EkfAutoAimContext`.
Don't reuse `ParticleFilterAutoAimContext` — it carries `sentry`/`hero`/
`standard`/`target_robot` fields that exist only for classification/targeting,
which you're dropping. An unused field is a "what is this for?" tax for the next
reader. Minimal fields:

```python
@dataclass
class EkfAutoAimContext(Context):
    start_loop_time: Optional[float] = None
    frame: Optional[np.ndarray] = None
    frame_ts: Optional[float] = None

    panels: Optional[List[ArmorPanel]] = None   # detection output
    new_observation: bool = False                # set after turret-frame transform

    estimate: Optional[RobotStateEstimate] = None
    solution: Optional[BallisticSolution] = None
```

`RobotStateEstimate` and `BallisticSolution` (`src/types/autoaim.py:87-118`) are
reused unchanged.

### Module list

```python
self.detection = ClassicalDetectorModule(self.ctx)              # outputs: panels
self.estimation = EkfEstimationModule(self.ctx)                  # NEW — inputs: panels, outputs: estimate
self.continuous_fire = FullStateContinuousFireModule(self.ctx)   # inputs: estimate, target_robot*
```

`FullStateContinuousFireModule` declares `inputs=["estimate", "target_robot"]`
(`src/subsystems/full_state_continuous_fire.py:201`), but `target_robot` doesn't
exist on `EkfAutoAimContext` — and you don't need it: `_run_solve` only checks
`if estimate is None or target_robot is None: return None` (line 219). It never
reads anything *out of* `target_robot`. Use `Module.remap_inputs`
(`src/core/module.py:187-192`) to point that input at `panels` instead:

```python
self.continuous_fire.remap_inputs(
    "full_state_continuous_fire", ["target_robot"], ["panels"]
)
```

Now `_validate_wiring()` sees `continuous_fire`'s inputs as `{"estimate",
"panels"}` — both produced elsewhere — and the `is None` check still does the
right thing (`panels` is `None` when nothing's detected). This is the same
"reuse through remapping" mechanism described in `docs/source/module.md` §5.

### `execute()` outline

```python
def execute(self) -> None:
    self.ctx.start_loop_time = time.perf_counter()

    # 0. newest frame (identical to PF engine, lines 122-128)
    frame = self.frames.latest()
    while frame is None or frame.seq == self._last_seq:
        time.sleep(0.0005)
        frame = self.frames.latest()
    self._last_seq = frame.seq
    self.ctx.frame = frame.data
    self.ctx.frame_ts = frame.timestamp

    # 1. detection
    self.detection.run()

    # 4. turret-frame transform (reuse helper verbatim — see §9)
    if self.ctx.panels:
        current_time = time.perf_counter()
        frame_delay_ms = int((current_time - self.ctx.frame_ts) * 1000)
        transformation_data = self.mcu.get_transformation(frame_delay_ms)
        if transformation_data is None:
            self.ctx.new_observation = False
        else:
            turret_yaw, _pitch, cam_to_turret = transformation_data
            _transform_panels_to_turret_frame(self.ctx.panels, cam_to_turret, turret_yaw)
            self.ctx.new_observation = True
    else:
        self.ctx.new_observation = False

    # 5. EKF estimation (always run — it predicts even with no observation)
    self.estimation.run()
    if self.ctx.estimate is None:
        return

    # 6. ballistics
    self.continuous_fire.run()
    if self.ctx.solution is None:
        return

    self._last_pitch = self.ctx.solution.pitch
    self._last_yaw = self.ctx.solution.yaw + np.deg2rad(config.ballistic.yaw_offset)
    self.alignment_time_ms = self.ctx.solution.alignment_time_ms  # always 255, continuous-fire
    self.cv_state = CVState.CONTINUOUS_FIRE.value
```

`update()` (sending the solution to MCU + display) can be copied verbatim from
`particle_filter_autoaim.py:227-238`.

---

## 7. The Tracking module: EKF spec

This is the new piece. New files: `src/subsystems/ekf.py` (the filter itself,
mirrors `src/subsystems/particle_filter.py`) and
`src/subsystems/ekf_estimation.py` (the `Module`, mirrors `src/subsystems/pf.py`).

### State vector — keep the existing convention

`FullStateContinuousFireModule._run_solve` consumes `estimate.value` as a 6-vector

```
x = [xc, yc, vx, vy, theta, omega]   # cm, cm, cm/s, cm/s, rad, rad/s
```

plus `estimate.a_radius` / `estimate.b_radius` for its 4-panel-at-90° geometry
(`_panel_normals`/`_select_panel`/`_cascade_solve`). **Keep this exact layout** so
`FullStateContinuousFireModule` needs zero changes.

`omega` is *known* (10 RPM = π/3 rad/s ≈ 1.047) and `R = a_radius = b_radius` is
*known*. Recommendation: **keep `omega` in the 6-D state**, just give it a tight
prior and tiny process noise — the filter then absorbs small real-world
deviations from exactly 10 RPM "for free," and the state vector stays the same
shape as the particle filter's, in case you ever want to compare them
side-by-side.

### Process model (predict step) — this part is linear

Constant velocity for the center, constant angular rate for the spin phase:

```
xc'    = xc + vx*dt
yc'    = yc + vy*dt
vx'    = vx
vy'    = vy
theta' = theta + omega*dt
omega' = omega
```

i.e. `x' = F @ x` with

```
F = [[1, 0, dt, 0, 0,  0],
     [0, 1, 0,  dt,0,  0],
     [0, 0, 1,  0, 0,  0],
     [0, 0, 0,  1, 0,  0],
     [0, 0, 0,  0, 1,  dt],
     [0, 0, 0,  0, 0,  1]]
```

`Q` (process noise) should be **small** for `omega` (we *know* the rate) and
small-to-moderate for `vx`/`vy` (the target may drift a bit). This mirrors
`_default_particle_filter`'s `Q_vel` (`src/subsystems/pf.py:27`), expressed as a
covariance instead of particle noise.

### Measurement model (update step)

Each detected `ArmorPanel` (after the turret-frame transform) gives an
observation `(x_obs, y_obs, yaw_obs)` — the same `[x, y, yaw]` triple the
particle filter consumes (`src/subsystems/pf.py:100-104`). Up to 4 panels can be
visible per frame; process them as independent sequential updates.

**Recommended approach — "back-projection"** (already implemented and tuned in
the particle filter's CUDA code, `src/subsystems/pf_cuda_cv.cu:191-232`,
`measurement_backproj`). For a panel at known radius `R` facing yaw `yaw_obs`,
its *center* is recoverable by a **linear** back-projection:

```
cx_obs = x_obs - R*cos(yaw_obs)
cy_obs = y_obs - R*sin(yaw_obs)
```

`(cx_obs, cy_obs)` is then a **direct, linear measurement of `(xc, yc)`**:

```
H_pos = [[1, 0, 0, 0, 0, 0],
         [0, 1, 0, 0, 0, 0]]
z_pos = [cx_obs, cy_obs]
```

For yaw, the target has 4-fold (π/2) rotational symmetry (`panel_ang = theta +
k*π/2` for `k=0..3`), so the *innovation* must be wrapped into `(-π/4, π/4]`
before the update (`yaw_residual` in the `.cu` file, lines 99-106):

```
innovation = ((theta - yaw_obs + π/4) mod π/2) - π/4
H_yaw = [0, 0, 0, 0, 1, 0]
```

Both `H_pos` and `H_yaw` are **constant matrices** — this is a **linear Kalman
filter** with a wrapped innovation for yaw, not a true nonlinear EKF. No
Jacobians needed. Start here.

**Alternative (a "real" EKF)**: measure each panel directly as `h(x) = [xc +
R*cos(theta), yc + R*sin(theta), wrap(theta)]` and linearize:

```
H = [[1, 0, 0, 0, -R*sin(theta), 0],
     [0, 1, 0, 0,  R*cos(theta), 0],
     [0, 0, 0, 0,  1,            0]]
```

recomputed every step from the current `theta`. Conceptually more "textbook," no
real benefit here since `R` is exact — mentioned for completeness if the
back-projection trick feels too clever.

### Initialization / reinit

Mirror `ParticleFilterEstimationModule._run_estimate`
(`src/subsystems/pf.py:76-122`):

- Track `is_their_target_prev: bool`. On the first frame a panel is seen
  (`False → True`), seed the state: `xc, yc` from the back-projected center (or
  raw panel position if using the direct EKF), `vx = vy = 0`, `theta =
  panel.yaw`, `omega = ω₀` (10 RPM in rad/s). Reset covariance `P` to a wide
  prior (large position/velocity uncertainty, *small* `omega` uncertainty).
- Reset `last_update_time` to `ctx.frame_ts` at reinit (avoids a stale/huge first
  `dt`, same as `pf.py:96`).
- If `panels` is `None`/empty: set `is_their_target_prev = False`, return `None`
  for `estimate` (matches `pf.py:80-82`).

### Predict-only path (no observation)

When `ctx.new_observation` is `False`, run the predict step only — `x' = F @ x`,
`P' = F P F^T + Q`, no measurement update. This is the EKF analogue of
`pf.update_with_no_observation(dt)` (`src/subsystems/pf.py:108-114`).

### `confidence`

`RobotStateEstimate.confidence` isn't used downstream beyond logging — a simple
proxy (`1 / (1 + trace(P_pos))`, or just `1.0`) is fine.

### Module I/O

```python
class EkfEstimationModule(Module[EkfAutoAimContext]):
    def __init__(self, context):
        super().__init__(
            name="ekf_estimation",
            context=context,
            inputs=["panels"],
            outputs=["estimate"],
        )
```

(Skipping the `target_robot`/`EnemyRobot` indirection entirely — with one target
there's nothing to wrap.)

### The `prediction(dt)` contract — required for `FullStateContinuousFireModule`

`FullStateContinuousFireModule._run_solve` calls
`self._pf.prediction(time_since_estimate)` (line 229) to advance the state by the
processing/feeder delay before solving, where `time_since_estimate =
time.perf_counter() - estimate.timestamp`. `set_particle_filter()`
(`full_state_continuous_fire.py:211-213`) just stores whatever object you pass as
`self._pf` — **it's duck-typed**. Give your EKF wrapper class one method:

```python
def prediction(self, dt: float) -> np.ndarray:
    """Constant-velocity/constant-omega extrapolation, no side effects."""
    pred = self.x.copy()  # current state estimate, shape (6,)
    pred[0] += pred[2] * dt
    pred[1] += pred[3] * dt
    pred[4] += pred[5] * dt
    return pred
```

This is exactly `pf_cuda_cv.cu`'s `pf_prediction` (lines 583-593) — copy it. Then
in the engine's `initialize()`:

```python
self.ekf = EkfTracker(radius=R, omega0=OMEGA0_RAD_S, ...)
self.estimation.set_estimator(self.ekf)             # whatever you name this setter
self.continuous_fire.set_particle_filter(self.ekf)  # name from base class; no change needed
```

`FullStateContinuousFireModule` requires **zero code changes**.

---

## 8. Learning Kalman / EKF filters

If Kalman filters are new to you, do these roughly in order. Don't try to derive
everything from first principles — get the intuition first, then map it onto the
state/process/measurement models in §7.

1. **YouTube — MATLAB Tech Talks: "Understanding Kalman Filters"** (search
   YouTube for that exact title; it's a ~16-part series by Brian Douglas /
   MathWorks). Episodes 1-5 cover the intuition and the predict/update cycle with
   simple 1-D and 2-D examples — that's the part you need. The later episodes
   cover the *Extended* Kalman filter (nonlinear `h(x)`), which is relevant if you
   go with the "alternative" measurement model in §7.

2. **Nice article that goes into more detail** https://wirelesspi.com/the-easiest-tutorial-on-kalman-filter/

3. Ask gemini to "make me a 1d kalman filter of a target with an oscillating position and show me a graph over time with noisy measurements and the kalman filter estimation and let me actively tune the measurement and process noise" 

### Using `filterpy`

`filterpy` is **already a dependency** of this project (see `pyproject.toml`),
so `from filterpy.kalman import KalmanFilter` just works under `uv run`.

A linear KF matching §7's recommended (back-projection) model looks like:

```python
import numpy as np
from filterpy.kalman import KalmanFilter

OMEGA0 = 10 * 2 * np.pi / 60  # 10 RPM -> rad/s

kf = KalmanFilter(dim_x=6, dim_z=2)  # state: [xc,yc,vx,vy,theta,omega]; meas: back-projected (cx,cy)

kf.x = np.array([0., 0., 0., 0., 0., OMEGA0])
kf.P = np.diag([100., 100., 50., 50., 1.0, 0.05])  # initial uncertainty (tune!)
kf.H = np.array([[1, 0, 0, 0, 0, 0],
                 [0, 1, 0, 0, 0, 0]])
kf.R = np.eye(2) * 20.0   # measurement noise (cm), tune from PF's r_pos
kf.Q = np.diag([1, 1, 5, 5, 0.01, 0.001])  # process noise, tune

def set_F(dt):
    kf.F = np.array([
        [1, 0, dt, 0, 0,  0],
        [0, 1, 0,  dt,0,  0],
        [0, 0, 1,  0, 0,  0],
        [0, 0, 0,  1, 0,  0],
        [0, 0, 0,  0, 1,  dt],
        [0, 0, 0,  0, 0,  1],
    ])

# each frame:
set_F(dt)
kf.predict()
if have_observation:
    cx_obs = x_obs - R * np.cos(yaw_obs)
    cy_obs = y_obs - R * np.sin(yaw_obs)
    kf.update(np.array([cx_obs, cy_obs]))
```

The yaw measurement doesn't fit `KalmanFilter.update()` directly because of the
π/2 wrap. Handle it as a *second*, 1-D update using `update()`'s `residual`
argument, which lets you supply your own `(measurement - prediction)` function
instead of plain subtraction:

```python
def yaw_residual(z, h_x):
    diff = (z - h_x + np.pi/4) % (np.pi/2) - np.pi/4
    return np.array([diff])

H_yaw = np.array([[0, 0, 0, 0, 1, 0]])
kf.update(np.array([yaw_obs]), H=H_yaw, R=np.array([[r_yaw**2]]), residual=yaw_residual)
```

(`filterpy`'s `update()` accepts per-call `H`/`R` overrides, which is convenient
here since you're alternating between a 2-D position update and a 1-D yaw
update.)

If you'd rather go the "alternative" route in §7 (true EKF, nonlinear `h(x)`),
look at `filterpy.kalman.ExtendedKalmanFilter` — same idea, but you supply
`HJacobian` (a function returning `H` evaluated at the current state) and `Hx` (a
function evaluating `h(x)`).

---

## 9. Reusing the turret-frame transform helper

`_transform_panels_to_turret_frame` (`src/engines/particle_filter_autoaim.py:247-264`)
is a free function, not a module — it mutates `panels` in place, converting
camera-relative `position`/`yaw` into the turret/ballistic frame using the 4x4
matrix + yaw returned by `mcu.get_transformation(frame_delay_ms)`. It's used
identically by both engines. Two options:

1. Copy it into the new engine file (simplest, fine for a first version).
2. Extract it to a shared module (e.g. `src/subsystems/turret_frame.py`) and
   import it from both engines.

Either is fine; don't over-engineer this for a first pass.

---

## 10. Reusing `ClassicalDetectorModule` — caveat

`ClassicalDetectorModule._process_pairs` (`src/subsystems/vision/classical_detector/module.py:25-65`)
calls `icon_detection.icon_detection(panel, frame)` and **drops any panel whose
icon doesn't match** one of the known RoboMaster armor digits
(`src/types/autoaim.py:58-71`, `ICON_TO_ROBOT_NAME`). If your spinning rig's
panels don't carry one of those recognized icons, **every panel will be filtered
out and `ctx.panels` will always be empty/`None`** — the EKF will never get a
measurement and you'll be stuck on the predict-only path forever, which can look
like "the EKF is broken" when really it's "detection is silently returning
nothing."

Resolve this before relying on detection:
- If the rig uses real armor-panel icons — no change needed.
- If it uses blank/different panels — either put real armor-panel icons on the
  rig, or add a config-gated bypass in `_process_pairs` (skip the
  `icon_detection` filter, set `panel.icon = None`).

A quick way to check which case you're in: temporarily set `config.log
.display_live_frames: true` and `@DISPLAY` and look at whether contours are drawn
around your panels at all.

---

## 11. Config additions (`src/info.yaml`)

```yaml
spinning_target:
    radius_cm: 23.5      # measure the actual rig; a_radius == b_radius == this
    omega_rpm: 10         # known constant spin rate
```

Convert once: `omega0 = config.spinning_target.omega_rpm * 2 * np.pi / 60`.
Sanity check: 10 RPM ≈ 1.047 rad/s, well under `ballistic.omega_spin_threshold`
(3.0 rad/s) — confirms continuous-fire is always correct for this target.

`config.ballistic.*` (gravity, projectile_velocity, z_offset, barrel_offset,
yaw_offset) is reused unchanged by `FullStateContinuousFireModule`.

---

## 12. Step-by-step: what to build, in order

1. **Add config** — put the `spinning_target` block (above) into
   `src/info.yaml` under `(default)`.

2. **Add the context** — add `EkfAutoAimContext` to `src/types/autoaim.py`
   (§6).

3. **Write the tracker** — `src/subsystems/ekf.py`: a small `EkfTracker` class
   with `__init__(radius, omega0, ...)`, `predict(dt)`, `update(x_obs, y_obs,
   yaw_obs)`, `reinit(prior)`, and `prediction(dt)` (§7). Get this working and
   sanity-checked on synthetic numbers (a quick `if __name__ == "__main__":`
   block feeding it a hand-computed circular trajectory and printing the
   estimate) *before* wiring it into a module — much faster to debug standalone.

4. **Write the module** — `src/subsystems/ekf_estimation.py`:
   `EkfEstimationModule(Module[EkfAutoAimContext])`, `inputs=["panels"]`,
   `outputs=["estimate"]`, `@real()` implementing the init/predict/update logic
   from §7, plus a `@mock` that returns a static `RobotStateEstimate` (e.g.
   `xc=0, yc=100, vx=vy=0, theta=0, omega=ω₀, a_radius=b_radius=R`) for testing
   without a camera.

5. **(Optional)** extract `_transform_panels_to_turret_frame` to
   `src/subsystems/turret_frame.py` (§9) — or just copy it for now.

6. **Write the engine** — `src/engines/spinning_target_autoaim.py`:
   - `drivers = {"frames": CameraDriver, "mcu": McuDriver}`
   - construct `self.ctx = EkfAutoAimContext()`, the three modules, call
     `self.continuous_fire.remap_inputs(...)` (§6)
   - `initialize()`: build `self.ekf = EkfTracker(...)`, call
     `self.estimation.set_estimator(self.ekf)` and
     `self.continuous_fire.set_particle_filter(self.ekf)`, get `self.frames =
     self.driver("frames")` and `self.mcu = self.driver("mcu")`
   - `execute()` / `update()` per §6.

7. **Add a launch block** at the bottom of the engine file, since it declares
   `drivers` and therefore must go through `launch_system` (§1):

   ```python
   if __name__ == "__main__":
       from src.core.orchestrator import launch_system
       processes = launch_system([SpinningTargetAutoAimEngine])
       try:
           for p in processes:
               p.join()
       except KeyboardInterrupt:
           for p in processes:
               p.terminate()
               p.join()
   ```

8. **Run it**:

   ```sh
   uv run python -m src.engines.spinning_target_autoaim @CAMERA=WEBCAM
   ```

   (`MCU=MOCK` is the default, so no turret hardware is required to see panels →
   estimate → ballistic solution flow end to end. Add `@DISPLAY` to see the
   annotated camera feed.)

9. **Iterate**:
   - No panels ever detected? → check §10 (icon filter).
   - `estimate` always `None`? → check the `is_their_target_prev` reinit logic
     and that `panels` isn't always `None`.
   - Estimate jumps around wildly? → `Q`/`R`/`P` tuning (start by making `R`
     bigger — i.e. trust measurements less — and see if it smooths out).
   - `solution` is `None`? → `FullStateContinuousFireModule` prints why
     (`"missing estimate or target_robot"` or `"solver failed"`) — check
     `estimate.value` looks sane (reasonable `xc`/`yc` in cm, `omega` near
     `OMEGA0`) before suspecting the solver.

---

## 13. Open design questions

1. **Keep `omega` in the state vector (6-D) with tiny `Q`/prior, or hardcode it
   (5-D state)?** 6-D recommended (§7) — trivial extra row/column, absorbs small
   real-world deviations from exactly 10 RPM.

2. **Back-projection linear KF vs. textbook EKF?** Back-projection (§7)
   recommended — it's what the existing, tuned particle filter already does, and
   it avoids Jacobians entirely.

3. **How many panels does the rig actually expose, and at what
   radius/spacing?** `FullStateContinuousFireModule`'s cascade solver
   (`_panel_normals`/`_select_panel`, `full_state_continuous_fire.py:70-96`)
   assumes 4 panels at 90° spacing (`a_radius` for panels 0/2, `b_radius` for
   panels 1/3). If the rig differs, that solver logic — not just the
   estimator — needs adjusting. Confirm the rig's geometry first.

---

## 14. Suggested file layout

```
src/types/autoaim.py
    + EkfAutoAimContext

src/subsystems/ekf.py            # EkfTracker: predict()/update()/prediction()
src/subsystems/ekf_estimation.py # EkfEstimationModule (Module[EkfAutoAimContext])
src/subsystems/turret_frame.py   # (optional) shared _transform_panels_to_turret_frame

src/engines/spinning_target_autoaim.py  # new Engine, mirrors particle_filter_autoaim.py
```
