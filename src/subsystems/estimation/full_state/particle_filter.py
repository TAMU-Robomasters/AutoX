"""Python wrapper around the pf_cuda_cv CUDA extension.

State vector (6-D):
    [x_c, y_c, vx, vy, theta, omega]

Measurement vector (3-D per panel):
    [x_panel, y_panel, panel_yaw]
"""

from __future__ import annotations

from typing import Optional

import numpy as np

try:
    import pf_cuda_cv as _ext
except ImportError as e:
    raise ImportError(
        "pf_cuda_cv extension not found. "
        "Run  uv sync  to compile the CUDA extension."
    ) from e


class ParticleFilter:
    """Constant-velocity bootstrap particle filter backed by native CUDA.

    Implements the ``FullStateEstimator`` protocol (see
    ``src/subsystems/estimation/filters.py``); its state is 6-D
    ``[x, y, vx, vy, theta, omega]`` (no height ``z``).
    """

    def __init__(
        self,
        num_particles: int,
        Q_vel: np.ndarray,       # (3,) process-noise std for [vx, vy, omega]
        r_pos: float,            # measurement noise std for panel position
        r_yaw: float,            # measurement noise std for panel yaw (rad)
        prior: np.ndarray,       # (6,) [x, y, vx, vy, theta, omega]
        init_std: np.ndarray,    # (6,) per-dim initial spread std
        radius: float = 23.5,
        compute_noise_stats: bool = False,
    ) -> None:
        self._compute_stats = compute_noise_stats

        _ext.init(
            num_particles,
            float(Q_vel[0]), float(Q_vel[1]), float(Q_vel[2]),
            float(r_pos), float(r_yaw), float(radius),
            prior.tolist(), init_std.tolist(),
            compute_noise_stats,
        )

        self.estimate = np.zeros(6, dtype=np.float32)

    def update(
        self,
        dt: float,
        measurements: np.ndarray,
        r: Optional[float] = None,
        z_meas: Optional[float] = None,
    ):
        """Motion + measurement + resample.

        measurements: np.ndarray shape (M, 3) — [[x, y, yaw], ...], M <= 4.
        Returns (estimate, confidence). ``r``/``z_meas`` are accepted for
        ``FullStateEstimator`` compatibility and ignored (the 6-D PF has a fixed
        orbit radius and no height state).
        """
        est_list, confidence = _ext.update(float(dt), np.ascontiguousarray(measurements, dtype=np.float32))
        self.estimate = np.array(est_list, dtype=np.float32)
        return self.estimate, confidence

    def update_with_no_observation(self, dt: float):
        """Motion only (no observation this frame)."""
        est_list, confidence = _ext.update_no_obs(float(dt))
        self.estimate = np.array(est_list, dtype=np.float32)
        return self.estimate, confidence

    def reinit(self, prior: np.ndarray) -> None:
        """Re-seed all particles around *prior* using the stored init_std."""
        _ext.reinit(prior[:6].tolist())
        self.estimate = np.zeros(6, dtype=np.float32)

    def accel(self) -> Optional[np.ndarray]:
        """The particle filter is constant-velocity only -- no acceleration state."""
        return None

    @staticmethod
    def predict_ahead(
        state: np.ndarray, dt: float, accel: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Constant-velocity extrapolation of a 6-D state ``dt`` seconds ahead.

        ``state``: ``[x, y, vx, vy, theta, omega]``. Pure function of the state
        (no particle cloud needed -- the CV mean extrapolation matches the native
        ``pf_prediction``), so the ballistic modules need no estimator instance.
        ``accel`` is accepted for the ``FullStateEstimator`` contract but ignored
        (the CUDA particle filter has no acceleration state).
        Satisfies ``FullStateEstimator.predict_ahead``.
        """
        out = np.array(state, dtype=np.float32).copy()
        out[0] = float(state[0]) + float(state[2]) * dt  # x += vx * dt
        out[1] = float(state[1]) + float(state[3]) * dt  # y += vy * dt
        out[4] = float(state[4]) + float(state[5]) * dt  # theta += omega * dt
        return out

    def noise_stats(self):
        """Return noise/uncertainty statistics dict (or None if disabled)."""
        if not self._compute_stats:
            return None
        return _ext.noise_stats()

    def enable_noise_stats(self, enabled: bool = True):
        """Enable/disable noise-stats computation."""
        self._compute_stats = bool(enabled)

    def __del__(self):
        try:
            _ext.destroy()
        except Exception:
            pass
