"""Synthetic tests for the EKF auto-aim engine pieces (no hardware).

* h(x) matches the original Full-Robot-EKF projection (frame mapping check)
* h_jacobian matches a brute-force finite difference
* the EKF converges on a spinning, translating robot seen through a moving gimbal
* the ported continuous-fire solver agrees with AutoX's solve_no_spin
* the RobotStateEstimate remap (units, aim_z, radii)
* the engine constructs (module wiring validates)
* two-panel seed, no update on the seed frame, 0.5 s lost reset
"""

import math
import sys
import types
from pathlib import Path

import cv2 as cv
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

sys.argv = [sys.argv[0]]  # before any config-touching src import

from src.subsystems.ballistics.ekf_continuous_fire import solve  # noqa: E402
from src.subsystems.ballistics.solver import solve_no_spin  # noqa: E402
from src.subsystems.estimation.ekf import model  # noqa: E402
from src.subsystems.estimation.ekf.filter import FullRobotEKF  # noqa: E402
from src.subsystems.estimation.ekf.model import (  # noqa: E402
    H, LOG_RA, LOG_RB, OMEGA, STATE_SIZE, THETA, VX, VY, X, Y, ZA,
)
from src.subsystems.estimation.ekf.module import to_robot_state_estimate  # noqa: E402
from src.types.autoaim import EkfStateEstimate  # noqa: E402

K = np.array([[900.0, 0, 640], [0, 900.0, 360], [0, 0, 1]])
DIST = np.array([-0.05, 0.02, 0.0005, -0.0003, 0.0])


def make_state(x=0.3, y=3.0, za=0.1, vx=0.0, vy=0.0, theta=0.4, omega=0.0,
               ra=0.22, rb=0.26, h=0.05):
    s = np.zeros(STATE_SIZE)
    s[[X, Y, ZA, VX, VY, THETA, OMEGA, H]] = [x, y, za, vx, vy, theta, omega, h]
    s[LOG_RA], s[LOG_RB] = np.log(ra), np.log(rb)
    return s


def extrinsic(yaw=0.0, pitch=0.0, offset=(0.0, 0.10, 0.05)):
    """camera->turret 4x4 (relabeled camera axes right/fwd/up -> turret), metres."""
    cz, sz = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    Rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
    M = np.eye(4)
    M[:3, :3] = Rz @ Rx
    M[:3, 3] = M[:3, :3] @ np.asarray(offset)
    return M


# ---------------------------------------------------------------------------
# Reference: the original repo's projection (camera frame, y down), trimmed copy
# of state_to_panels_corner_2D.state_to_panel_corners_2d @5672b45.
# ---------------------------------------------------------------------------

def _reference_corners(x, y, z, theta, a, b, h):
    hw, hh, lw = 0.137 / 2, 0.057 / 2, 0.008
    local = np.array([
        [(hw - lw, -hh), (hw, -hh), (hw, hh), (hw - lw, hh)],     # right light
        [(-hw, -hh), (-hw + lw, -hh), (-hw + lw, hh), (-hw, hh)],  # left light
    ])
    radials = np.array([(0, -1), (1, 0), (0, 1), (-1, 0)], dtype=float)
    ct, st = math.cos(theta), math.sin(theta)
    centers = np.array([
        (x - a * st, y, z - a * ct), (x + b * ct, y - h, z - b * st),
        (x + a * st, y, z + a * ct), (x - b * ct, y - h, z + b * st),
    ])
    nx = radials[:, 0] * ct + radials[:, 1] * st
    nz = -radials[:, 0] * st + radials[:, 1] * ct
    tilt = math.radians(15.0)
    up = np.stack([-math.sin(tilt) * nx, np.full(4, -math.cos(tilt)), -math.sin(tilt) * nz], 1)
    right = np.stack([-nz, np.zeros(4), nx], 1)
    pts = (centers[:, None, None] + local[None, :, :, 0, None] * right[:, None, None]
           - local[None, :, :, 1, None] * up[:, None, None])
    uv, _ = cv.projectPoints(pts.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), K, DIST)
    return uv.reshape(4, 2, 4, 2)  # [panel, (right, left), corner, uv]


def test_h_matches_original_repo_projection():
    """Identity extrinsic: turret (x, y, z) == camera (x_cv, z_cv, -y_cv).

    Original theta_cv maps to ours as theta = -pi/2 - theta_cv with the same
    panel indices; the original lists lights (right, left), ours (left, right).
    """
    xc, yc, zc, th_cv, a, b, hh = 0.2, -0.1, 3.0, 0.3, 0.22, 0.26, 0.04
    ref = _reference_corners(xc, yc, zc, th_cv, a, b, hh)
    s = make_state(x=xc, y=zc, za=-yc, theta=-math.pi / 2 - th_cv, ra=a, rb=b, h=hh)
    cam = model.Camera(K, DIST, np.eye(4))
    ours = model.h(s, cam, [0, 1, 2, 3]).reshape(4, 2, 4, 2)
    np.testing.assert_allclose(ours[:, 0], ref[:, 1], atol=1e-6)
    np.testing.assert_allclose(ours[:, 1], ref[:, 0], atol=1e-6)


def test_h_jacobian_matches_finite_difference():
    s = make_state()
    cam = model.Camera(K, DIST, extrinsic(yaw=0.1, pitch=0.05))
    J = model.h_jacobian(s, cam, [0, 1])
    ref = np.zeros_like(J)
    for i in range(STATE_SIZE):
        d = np.zeros(STATE_SIZE)
        d[i] = 1e-6
        ref[:, i] = (model.h(s + d, cam, [0, 1]) - model.h(s - d, cam, [0, 1])) / 2e-6
    np.testing.assert_allclose(J, ref, rtol=1e-3, atol=1e-2)


# ---------------------------------------------------------------------------
# EKF convergence on synthetic data
# ---------------------------------------------------------------------------


def _facing_panels(s, M, max_angle_deg=60.0):
    cam_pos = M[:3, 3]
    centers = model.panel_centers(s)
    out = []
    for k in range(4):
        phi = s[THETA] + k * math.pi / 2
        n = np.array([math.cos(phi), math.sin(phi), 0.0])
        to_cam = cam_pos - centers[k]
        cosang = n @ to_cam / np.linalg.norm(to_cam)
        if cosang > math.cos(math.radians(max_angle_deg)):
            out.append(k)
    return out


def _run_sim(truth0, seconds=4.0, fps=90.0, px_noise=1.0, seed=0):
    rng = np.random.default_rng(seed)
    ekf = FullRobotEKF()
    truth = truth0.copy()
    dt = 1.0 / fps
    seeded_k = None
    vel_err = []  # (vx, vy) estimate - truth per frame (velocity is label-invariant)
    for i in range(int(seconds * fps)):
        t = i * dt
        if i:
            truth = model.f(truth, dt)
        # gimbal roughly tracks the robot with a ripple, so the extrinsic changes every frame
        yaw = math.atan2(truth[Y], truth[X]) - math.pi / 2 + 0.05 * math.sin(7 * t)
        M = extrinsic(yaw=yaw, pitch=0.03 * math.sin(5 * t))
        cam = model.Camera(K, DIST, M)
        vis = _facing_panels(truth, M)
        if not vis:
            continue
        dets = [model.h(truth, cam, [k]).reshape(2, 4, 2) + rng.normal(0, px_noise, (2, 4, 2))
                for k in vis]
        if not ekf.initialized:
            k0 = min(vis, key=lambda k: np.linalg.norm(model.panel_centers(truth)[k]))
            pos = model.panel_centers(truth)[k0] + rng.normal(0, 0.02, 3)
            yaw_k = truth[THETA] + k0 * math.pi / 2 + rng.normal(0, math.radians(5))
            ekf.seed(pos, yaw_k)
            seeded_k = k0
        else:
            ekf.predict(dt)
        ekf.update(dets, cam)
        vel_err.append(ekf.x[[VX, VY]] - truth[[VX, VY]])
    _run_sim.last_vel_err = np.array(vel_err)
    return ekf, truth, seeded_k


def _expected_in_ekf_labels(truth, k0):
    """Truth re-expressed with the seeded panel k0 relabeled as panel 0."""
    e = truth.copy()
    e[THETA] = model.wrap_angle(truth[THETA] + k0 * math.pi / 2)
    if k0 % 2:
        e[LOG_RA], e[LOG_RB] = truth[LOG_RB], truth[LOG_RA]
        e[ZA] = truth[ZA] + truth[H]
        e[H] = -truth[H]
    return e


@pytest.mark.parametrize("omega", [0.0, 4.0])
def test_ekf_converges_on_moving_robot(omega):
    truth0 = make_state(x=0.3, y=3.0, za=0.1, vx=0.3, vy=-0.1, theta=0.4,
                        omega=omega, ra=0.24, rb=0.26, h=0.03)
    ekf, truth, k0 = _run_sim(truth0)
    exp = _expected_in_ekf_labels(truth, k0)
    x = ekf.x
    assert ekf.accepted_count > 300
    assert np.hypot(x[X] - exp[X], x[Y] - exp[Y]) < 0.03, (x[:3], exp[:3])
    assert abs(x[ZA] - exp[ZA]) < 0.01
    assert abs(x[OMEGA] - exp[OMEGA]) < 0.1
    assert abs(model.wrap_angle(x[THETA] - exp[THETA])) < math.radians(3)
    # The process noise lets velocity wander (~10 cm/s per frame by design: it
    # must follow dodging targets), so check the last second's mean -> no bias.
    vel_bias = _run_sim.last_vel_err[-90:].mean(axis=0)
    assert np.hypot(*vel_bias) < 0.05, vel_bias
    assert abs(x[H] - exp[H]) < 0.005
    if omega:  # both panel pairs get seen -> radii become observable
        assert abs(math.exp(x[LOG_RA]) - math.exp(exp[LOG_RA])) < 0.005
        assert abs(math.exp(x[LOG_RB]) - math.exp(exp[LOG_RB])) < 0.005


# ---------------------------------------------------------------------------
# Ballistics
# ---------------------------------------------------------------------------


def test_seed_pair_recovers_centre_radii_height():
    truth = make_state(x=0.4, y=2.3, za=0.1, theta=0.3, ra=0.22, rb=0.27, h=0.03)
    c = model.panel_centers(truth)
    for kb in (1, 3):                      # either odd neighbour of panel 0
        ekf = FullRobotEKF()
        assert ekf.seed_pair(c[0], truth[THETA], c[kb], truth[THETA] + kb * math.pi / 2)
        x = ekf.x
        assert np.allclose(x[[X, Y, ZA, H]], truth[[X, Y, ZA, H]], atol=1e-9)
        assert abs(model.wrap_angle(x[THETA] - truth[THETA])) < 1e-9
        assert np.allclose(np.exp(x[[LOG_RA, LOG_RB]]), [0.22, 0.27], atol=1e-9)
        assert ekf.initialized and ekf.accepted_count == 0
    # opposite panels (180 deg apart) are not an adjacent pair -> untouched
    ekf = FullRobotEKF()
    assert not ekf.seed_pair(c[0], truth[THETA], c[2], truth[THETA] + math.pi)
    assert not ekf.initialized


def _module_and_panels(truth, ks, M):
    from src.subsystems.estimation.ekf.module import EkfEstimationModule

    mod = EkfEstimationModule(types.SimpleNamespace(), K, DIST)
    cam = model.Camera(K, DIST, M)
    c = model.panel_centers(truth)
    panels = [types.SimpleNamespace(position=c[k] * 100.0, yaw=truth[THETA] + k * math.pi / 2,
                                    light_corners=model.h(truth, cam, [k]).reshape(2, 4, 2))
              for k in ks]
    return mod, panels


def test_module_seeds_without_update_then_resets_when_lost():
    from src.subsystems.estimation.ekf.module import LOST_TIME_S

    truth = make_state(x=0.3, y=2.3, za=0.1, theta=0.6, ra=0.22, rb=0.27, h=0.03)
    M = extrinsic(yaw=math.atan2(truth[Y], truth[X]) - math.pi / 2)
    mod, panels = _module_and_panels(truth, [0, 1], M)
    mod._observe(panels, 0.0, M)                                   # seed frame
    assert mod.seed_mode == "two_panels" and mod.ekf.accepted_count == 0
    mod._observe(panels, 0.01, M)
    assert mod.ekf.accepted_count == 1

    # Detections far from the prediction keep getting gated out ...
    moved = make_state(x=-0.8, y=3.5, za=0.1, theta=-0.2, ra=0.22, rb=0.27, h=0.03)
    _, far = _module_and_panels(moved, [0], M)
    far[0].light_corners = far[0].light_corners + 700.0           # beyond the 600 px init gate
    for i in range(1, int(LOST_TIME_S / 0.02)):                   # up to 0.48 s rejected
        mod._observe(far, 0.01 + 0.02 * i, M)
        assert mod.n_resets == 0
    # ... until LOST_TIME_S without an accepted update -> reset + re-seed here.
    mod._observe(far, 0.01 + LOST_TIME_S + 0.02, M)
    assert mod.n_resets == 1 and mod.seed_mode == "one_panel"
    assert np.allclose(mod.ekf.x[[X, Y]] + 0.25 * np.array([math.cos(-0.2), math.sin(-0.2)]),
                       model.panel_centers(moved)[0][:2], atol=1e-9)

def test_solver_matches_solve_no_spin_for_nonspinning_target():
    """No spin, zero barrel offset: the ported solver is exact no-drag ballistics."""
    s = make_state(x=0.5, y=4.0, za=0.2, vx=0.4, vy=0.2, theta=-math.pi / 2,
                   omega=0.0, ra=0.25, rb=0.25, h=0.0)
    shot = solve(s, speed=28.0, g=9.81, t_max=1.5)
    assert shot is not None and shot.panel == 0  # panel 0 faces the turret (normal -y)
    p_panel = model.panel_centers(s)[0]
    ref = solve_no_spin(np.zeros(3), 28.0, -9.81, p_panel, np.array([0.4, 0.2, 0.0]))
    assert ref["success"]
    assert abs(shot.yaw - ref["yaw"]) < 1e-3
    assert abs(shot.pitch - ref["pitch"]) < 1e-3
    assert abs(shot.t_flight - ref["time"]) < 1e-3


def test_solver_yaw_convention_and_gravity():
    """Dead ahead (+y) is MCU yaw 0; target to the right is negative yaw; aims above."""
    ahead = solve(make_state(x=0.0, y=5.0, za=0.0, theta=-math.pi / 2, ra=0.25, rb=0.25, h=0.0),
                  speed=28.0)
    assert abs(ahead.yaw) < 1e-6
    assert ahead.pitch > 0  # gravity compensation on a level target
    right = solve(make_state(x=2.0, y=5.0, za=0.0, theta=math.pi, ra=0.25, rb=0.25, h=0.0),
                  speed=28.0)
    assert right is not None and right.yaw < 0


def test_remap_to_robot_state_estimate():
    s = make_state(x=0.3, y=3.0, za=0.1, vx=0.5, vy=-0.2, theta=0.4, omega=2.0,
                   ra=0.22, rb=0.26, h=0.04)
    est = to_robot_state_estimate(EkfStateEstimate(x=s, timestamp=12.5))
    np.testing.assert_allclose(est.value, [30, 300, 50, -20, 0.4, 2.0, 12.0], atol=1e-9)
    assert est.aim_z == pytest.approx(12.0)
    assert est.a_radius == pytest.approx(22.0)
    assert est.b_radius == pytest.approx(26.0)
    assert est.timestamp == 12.5


# ---------------------------------------------------------------------------
# Engine wiring + detector corner order (own interpreter: needs the @SENTRY
# profile for camera dimensions, and stubs for mrcal / the legacy camera stream)
# ---------------------------------------------------------------------------

_SUBPROCESS_CHECK = r"""
import sys, types, numpy as np, cv2 as cv
sys.argv = ["x", "@SENTRY"]
sys.modules["mrcal"] = types.ModuleType("mrcal")
legacy = types.ModuleType("src.subsystems.video_streaming.video_stream")
legacy.video_stream = None
sys.modules["src.subsystems.video_streaming.video_stream"] = legacy
import src.drivers.video_stream as vs
K = np.array([[900.0, 0, 640], [0, 900.0, 360], [0, 0, 1]])
vs._CameraInfo.intrinsics = lambda self: vs.Intrinsics(distortion_coefficients=np.zeros(5), camera_matrix=K)

from src.engines.ekf_autoaim import EkfAutoAimEngine
engine = EkfAutoAimEngine()
names = [m.name for m in engine.modules]
assert names == ["classical_detector", "robot_classification", "targeting",
                 "ekf_estimation", "ekf_continuous_fire", "full_state_shot_timing"], names

# Detector light corners (minAreaRect -> Lights.corners) line up with h()'s order.
from src.subsystems.estimation.ekf import model
from src.subsystems.vision.classical_detector import armor
s = np.zeros(model.STATE_SIZE)
s[[model.X, model.Y, model.ZA, model.THETA]] = [0.1, 2.5, 0.0, -np.pi / 2 + 0.3]
s[model.LOG_RA] = s[model.LOG_RB] = np.log(0.25)
cam = model.Camera(K, np.zeros(5), np.eye(4))
pred = model.h(s, cam, [0]).reshape(2, 4, 2)
lights = []
for light in pred[::-1]:  # feed right-then-left: light_corners must sort by x
    rect = cv.minAreaRect(light.astype(np.float32))
    lights.append(armor.Lights(rect[0][0], rect[0][1], 0, 0, 0, rect=rect))
got = armor.light_corners(lights)
err = np.abs(got - pred).max()
assert err < 1.5, (err, got, pred)

# Corner order is a real rectangle at any lean, including exactly 45 deg where
# the old x+y / y-x shortcut picked one corner twice ("triangle" light).
for lean in (-45.0, -20.0, 0.0, 20.0, 45.0, 90.0):
    for w, h in ((8, 40), (40, 8)):
        box = cv.boxPoints(((300.0, 200.0), (w, h), lean)).astype(np.float64)
        tl, tr, br, bl = armor.order_corners(box)
        assert len({tuple(np.round(p, 6)) for p in (tl, tr, br, bl)}) == 4, (lean, w, h)
        top, bottom = (tl + tr) / 2, (bl + br) / 2
        axis = top - bottom
        assert np.linalg.norm(axis) > 30 and axis[1] <= 0, (lean, w, h, axis)  # long axis, up (or flat)
        across = tr - tl
        cos = across @ axis / (np.linalg.norm(across) * np.linalg.norm(axis))
        assert abs(cos) < 1e-3, (lean, w, h, cos)                                 # square corners
        assert axis[0] * across[1] - axis[1] * across[0] > 0, (lean, w, h)         # tr right of tl
print("OK", err)
"""


def test_engine_wiring_and_detector_corner_order():
    import subprocess

    result = subprocess.run(
        [sys.executable, "-c", _SUBPROCESS_CHECK],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0 and "OK" in result.stdout, result.stderr[-3000:]
