"""Motion-model strategies for the centre/panel position filter (CV vs CA).

The 2-D position tracker (``PositionKF`` in ``filters.py``) has an interchangeable
*dynamics* half and a fixed *measurement* half. This module owns the dynamics:
the internal state dimension, the transition matrix ``F(dt)``, the process-noise
matrix ``Q(dt)``, the seed state/covariance, and how to read the (optional)
acceleration back out. The measurement half (``back_project`` / ``H`` / ``R``) is
identical for both models and stays in ``PositionKF``.

We use **composition** rather than inheritance: CV and CA differ *only* in the
dynamics, while the measurement update, the per-measurement back-projection
radius, and the published ``[x, y, vx, vy]`` slice are shared. A motion-model
object is swapped into the one ``PositionKF`` (and is selected from
``config.estimation.motion_model``); subclassing ``PositionKF`` would instead
force the shared measurement code into a base class, multiply the class count,
and -- because ``predict_ahead`` is a static, instance-free extrapolator the
ballistic modules call off a bare state array -- still could not dispatch by
subtype. The angle and height filters are always constant-velocity, so they get
no model object at all.

Both models keep the published state layout ``[x, y, vx, vy]`` as the first four
internal entries, so every downstream consumer (``FullStateKF``'s 7-D layout,
ballistics) is unaffected by the model choice. The acceleration is appended
(CA: ``[x, y, vx, vy, ax, ay]``) and surfaced separately via ``accel`` -- never
inserted into the middle of the vector -- so the index-based consumers stay
valid.

All units are the estimator's working units (cm, cm/s, cm/s^2, s).
"""

from __future__ import annotations

from typing import Optional, Protocol

import numpy as np

CONSTANT_VELOCITY = "constant_velocity"
CONSTANT_ACCELERATION = "constant_acceleration"
# Third ``config.estimation.motion_model`` value. The IMM is NOT a MotionModel
# (it runs several models at once) -- it is built by ``imm.make_position_estimator``
# as a separate estimator, not via ``make_motion_model`` below.
IMM = "imm"


class MotionModel(Protocol):
    """Dynamics strategy for ``PositionKF`` (the only CV/CA switch in the stack).

    Implementations are pure dynamics: no measurement, no filtering state. The
    first four entries of every state vector are always ``[x, y, vx, vy]``.
    """

    dim: int
    """Internal state dimension (4 for CV, 6 for CA)."""

    def transition(self, dt: float) -> np.ndarray:
        """State-transition matrix ``F`` for a step of ``dt`` seconds."""
        ...

    def process_noise(self, dt: float) -> np.ndarray:
        """Process-noise covariance ``Q`` for a step of ``dt`` seconds."""
        ...

    def seed_state(self, x0: float, y0: float, vx0: float, vy0: float) -> np.ndarray:
        """Initial state vector (acceleration, if tracked, starts at 0)."""
        ...

    def seed_cov(self, init_std: np.ndarray) -> np.ndarray:
        """Initial covariance from the 4-vector ``init_std`` of ``[x,y,vx,vy]`` stds."""
        ...

    def accel(self, state: np.ndarray) -> Optional[np.ndarray]:
        """The ``[ax, ay]`` acceleration in ``state``; ``None`` for CV."""
        ...


class ConstantVelocity:
    """``[x, y, vx, vy]`` constant-velocity model (white-noise-acceleration Q).

    ``q_vx`` / ``q_vy`` are the per-axis white-noise-acceleration spectral
    densities (the noise enters at the velocity level).
    """

    dim = 4

    def __init__(self, q_vx: float = 100.0, q_vy: float = 100.0) -> None:
        self.q_vx = q_vx
        self.q_vy = q_vy

    def transition(self, dt: float) -> np.ndarray:
        """``F`` for ``[x, y, vx, vy]``: position advances by ``v*dt``."""
        return np.array(
            [
                [1.0, 0.0, dt, 0.0],
                [0.0, 1.0, 0.0, dt],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )

    def process_noise(self, dt: float) -> np.ndarray:
        """White-noise-acceleration ``Q`` driven per axis by ``q_vx``/``q_vy``."""
        qx, qy = self.q_vx**2, self.q_vy**2
        return np.array(
            [
                [qx * dt**3 / 3.0, 0.0, qx * dt**2 / 2.0, 0.0],
                [0.0, qy * dt**3 / 3.0, 0.0, qy * dt**2 / 2.0],
                [qx * dt**2 / 2.0, 0.0, qx * dt, 0.0],
                [0.0, qy * dt**2 / 2.0, 0.0, qy * dt],
            ]
        )

    def seed_state(self, x0: float, y0: float, vx0: float, vy0: float) -> np.ndarray:
        """``[x0, y0, vx0, vy0]``."""
        return np.array([x0, y0, vx0, vy0], dtype=np.float64)

    def seed_cov(self, init_std: np.ndarray) -> np.ndarray:
        """Diagonal covariance from the 4-vector of ``[x,y,vx,vy]`` stds."""
        return np.diag(np.asarray(init_std, dtype=np.float64) ** 2)

    def accel(self, state: np.ndarray) -> Optional[np.ndarray]:
        """CV has no acceleration state."""
        return None


class ConstantAcceleration:
    """``[x, y, vx, vy, ax, ay]`` constant-acceleration model (white-noise-jerk Q).

    The acceleration is tracked but the **only** process-noise driver is jerk
    (``q_jerk``): the position and velocity process noise are *derived* from it
    via the standard continuous white-noise-jerk Q. Lower bandwidth (smoother)
    than CV at the cost of some ringing -- the intended A/B against CV.
    """

    dim = 6

    def __init__(self, q_jerk: float = 30.0, init_std_accel: float = 50.0) -> None:
        self.q_jerk = q_jerk
        self.init_std_accel = init_std_accel

    def transition(self, dt: float) -> np.ndarray:
        """``F`` for ``[x, y, vx, vy, ax, ay]`` (block-ordered pos, vel, accel)."""
        # Block-ordered (pos, vel, accel) so the first four entries stay [x,y,vx,vy].
        F = np.eye(6)
        F[0, 2] = dt
        F[1, 3] = dt
        F[2, 4] = dt
        F[3, 5] = dt
        F[0, 4] = 0.5 * dt**2
        F[1, 5] = 0.5 * dt**2
        return F

    def process_noise(self, dt: float) -> np.ndarray:
        """Continuous white-noise-jerk ``Q``: a single ``q_jerk`` driver per axis."""
        q = self.q_jerk**2
        d5, d4, d3, d2 = dt**5, dt**4, dt**3, dt**2
        Q = np.zeros((6, 6))
        # Per axis the [p, v, a] white-noise-jerk block; x uses (0,2,4), y (1,3,5).
        for ip, iv, ia in ((0, 2, 4), (1, 3, 5)):
            Q[ip, ip] = q * d5 / 20.0
            Q[ip, iv] = Q[iv, ip] = q * d4 / 8.0
            Q[ip, ia] = Q[ia, ip] = q * d3 / 6.0
            Q[iv, iv] = q * d3 / 3.0
            Q[iv, ia] = Q[ia, iv] = q * d2 / 2.0
            Q[ia, ia] = q * dt
        return Q

    def seed_state(self, x0: float, y0: float, vx0: float, vy0: float) -> np.ndarray:
        """``[x0, y0, vx0, vy0, 0, 0]`` (acceleration starts at rest)."""
        return np.array([x0, y0, vx0, vy0, 0.0, 0.0], dtype=np.float64)

    def seed_cov(self, init_std: np.ndarray) -> np.ndarray:
        """Diagonal covariance: ``init_std`` for pos/vel, ``init_std_accel`` for accel."""
        var = np.concatenate(
            [np.asarray(init_std, dtype=np.float64) ** 2, [self.init_std_accel**2] * 2]
        )
        return np.diag(var)

    def accel(self, state: np.ndarray) -> Optional[np.ndarray]:
        """The filtered ``[ax, ay]`` from the state."""
        return np.array([float(state[4]), float(state[5])], dtype=np.float64)


def make_motion_model(
    model: str,
    *,
    q_vx: float = 100.0,
    q_vy: float = 100.0,
    q_jerk: float = 30.0,
    init_std_accel: float = 50.0,
) -> MotionModel:
    """Build the motion model named by ``model`` (the ``config.estimation.motion_model`` string)."""
    if model == CONSTANT_VELOCITY:
        return ConstantVelocity(q_vx=q_vx, q_vy=q_vy)
    if model == CONSTANT_ACCELERATION:
        return ConstantAcceleration(q_jerk=q_jerk, init_std_accel=init_std_accel)
    raise ValueError(
        f"motion_model must be {CONSTANT_VELOCITY!r} or {CONSTANT_ACCELERATION!r}, got {model!r}"
    )
