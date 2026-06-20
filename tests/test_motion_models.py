"""CV/CA motion-model composition + acceleration-aware prediction and solving.

Config-free on purpose: imports only ``motion_models``, ``filters``, and the
ballistic ``solver`` (none of which read ``info.yaml``), so it exercises the
acceleration plumbing without the engine/config stack.
"""

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.subsystems.ballistics.solver import make_f_no_spin, solve_no_spin  # noqa: E402
from src.subsystems.estimation.filters import FullStateKF, PositionKF  # noqa: E402
from src.subsystems.estimation.motion_models import (  # noqa: E402
    ConstantAcceleration,
    ConstantVelocity,
    make_motion_model,
)


def test_model_dims_and_accel_readback():
    """CV is 4-D with no accel; CA is 6-D and reads [ax, ay] back from the state."""
    cv = make_motion_model("constant_velocity", q_vx=50, q_vy=50)
    ca = make_motion_model("constant_acceleration", q_jerk=100, init_std_accel=200)
    assert cv.dim == 4 and ca.dim == 6
    assert cv.accel(np.zeros(4)) is None
    np.testing.assert_allclose(ca.accel(np.array([0, 0, 0, 0, 3.0, 4.0])), [3, 4])


def test_make_motion_model_rejects_unknown():
    """An unrecognised model name raises ValueError."""
    import pytest

    with pytest.raises(ValueError):
        make_motion_model("constant_jerk")


def test_position_kf_accel_none_for_cv_vector_for_ca():
    """PositionKF.accel() is None under a CV model and a 2-vector under CA."""
    cv = PositionKF(motion_model=ConstantVelocity(1, 1), r_pos=1, r=0, init_std=(50, 50, 5, 5))
    cv.reinit(0, 0)
    assert cv.accel() is None
    ca = PositionKF(
        motion_model=ConstantAcceleration(100, 200), r_pos=1, r=0, init_std=(50, 50, 5, 5)
    )
    ca.reinit(0, 0)
    assert ca.accel() is not None and ca.accel().shape == (2,)


def test_predict_ahead_cv_vs_ca():
    """predict_ahead is CV with accel=None and folds 0.5*a*t^2 when accel is given."""
    s = np.array([0.0, 0.0, 10.0, 0.0])  # x, y, vx, vy
    # accel=None -> constant velocity
    np.testing.assert_allclose(PositionKF.predict_ahead(s, 2.0), [20, 0, 10, 0])
    # accel given -> x += v*t + 0.5*a*t^2, v += a*t  (8 cm, 8 cm/s over 2 s @ 4)
    np.testing.assert_allclose(
        PositionKF.predict_ahead(s, 2.0, np.array([4.0, 0.0])), [28, 0, 18, 0]
    )


def test_full_state_predict_ahead_threads_accel_and_omega():
    """The 7-D predict_ahead applies accel to position, omega to theta, and holds z."""
    fs = np.array([0, 0, 10, 0, 0.0, 1.0, 50.0])  # ..., theta, omega, z
    out = FullStateKF.predict_ahead(fs, 2.0, np.array([4.0, 0.0]))
    np.testing.assert_allclose(out[:4], [28, 0, 18, 0])
    np.testing.assert_allclose(out[4], 2.0)  # theta += omega * dt
    np.testing.assert_allclose(out[6], 50.0)  # z held


def test_gate_radius_includes_q_and_shrinks_with_updates():
    """gate_radius grows with horizon (the +Q term) and shrinks as the track tightens."""
    kf = PositionKF(motion_model=ConstantVelocity(50, 50), r_pos=3, r=0, init_std=(50, 50, 50, 50))
    kf.reinit(0, 0)
    wide = kf.gate_radius(0.14)
    # Longer horizon -> more predicted spread (Q accumulates).
    assert kf.gate_radius(0.5) > kf.gate_radius(0.14)
    # Feed consistent observations at the origin; the gate radius must tighten.
    for _ in range(30):
        kf.update(1 / 90.0, 0.0, 0.0, 0.0)
    assert kf.gate_radius(0.14) < wide


def test_solver_a_t_none_matches_zeros_and_hits_accelerating_target():
    """a_t=None equals a_t=0, and a non-zero a_t yields a zero-residual solution."""
    b, spd, g = np.zeros(3), 28.0, -9.81
    p, v = np.array([0.0, 5.0, 0.0]), np.zeros(3)
    none_sol = solve_no_spin(b, spd, g, p, v)
    zero_sol = solve_no_spin(b, spd, g, p, v, a_t=np.zeros(3))
    assert none_sol["success"] and zero_sol["success"]
    for k in ("yaw", "pitch", "time"):
        assert none_sol[k] == np.float64(zero_sol[k]) or abs(none_sol[k] - zero_sol[k]) < 1e-9
    # Accelerating target: the residual at the returned solution is ~0.
    a = np.array([3.0, 0.0, 0.0])
    sol = solve_no_spin(b, spd, g, p, v, a_t=a)
    assert sol["success"]
    f = make_f_no_spin(b, spd, g, p, v, a)
    assert np.linalg.norm(f(np.array([sol["yaw"], sol["pitch"], sol["time"]]))) < 1e-4
