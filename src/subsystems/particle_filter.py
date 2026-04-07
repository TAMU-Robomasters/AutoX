"""GPU-accelerated particle filter for tracking a spinning robot.

Ported from Armor-Panel-Classical/subsystems/pf.py. Uses CuPy for
GPU computation

State vector (7-D):
    [x_c, y_c, vx, vy, theta, omega, radius]

Measurement vector (3-D):
    [x_panel, y_panel, panel_yaw]
"""

import cupy as cp


class ParticleFilter:
    """Bootstrap particle filter with systematic resampling."""

    def __init__(
        self,
        num_particles: int,
        Q: cp.ndarray,
        R: cp.ndarray,
        prior: cp.ndarray,
        num_meas: int,
        compute_noise_stats: bool = False,
    ):
        self.num_particles = num_particles
        self.num_meas = num_meas
        self.num_dimensions = len(Q)

        # ---- Covariances (float32) ----
        self.Q = Q.astype(cp.float32)
        self.R = R.astype(cp.float32)

        # Assume diagonal Q and R for max performance
        self.Q_sqrt_diag = cp.sqrt(cp.diag(self.Q)).astype(cp.float32)
        self.R_inv_diag = (1.0 / cp.diag(self.R)).astype(cp.float32)

        # ---- Particle state ----
        self.xk = cp.empty((num_particles, self.num_dimensions), dtype=cp.float32)

        # Preallocate noise buffer
        self.noise = cp.empty_like(self.xk)

        # Initialize particles
        self.noise[:] = cp.random.standard_normal(self.noise.shape, dtype=cp.float32)
        self.noise *= self.Q_sqrt_diag
        self.xk[:] = prior + self.noise

        # ---- Weights ----
        self.weights = cp.full(num_particles, 1.0 / num_particles, dtype=cp.float32)

        # ---- Buffers ----
        self.resample_buffer = cp.empty_like(self.xk)
        self.indices_base = cp.arange(num_particles, dtype=cp.int32)
        self.zk = cp.empty((num_particles, self.num_meas), dtype=cp.float32)

        # ---- Threshold ----
        self.thresh = num_particles / 3

        # ---- Initial estimate ----
        self.estimate = cp.mean(self.xk, axis=0)

        # ---- Noise / uncertainty stats (updated each `update`) ----
        # Velocity components: xk[:,2]=vx, xk[:,3]=vy. Angular velocity: xk[:,5]=omega.
        self.noise_stats_enabled = bool(compute_noise_stats)
        self.vx_std = cp.float32(0.0)
        self.vy_std = cp.float32(0.0)
        self.speed_std = cp.float32(0.0)
        self.omega_std = cp.float32(0.0)

    def enable_noise_stats(self, enabled: bool = True, reset: bool = False):
        """Enable or disable noise statistics tracking."""
        self.noise_stats_enabled = bool(enabled)
        if reset and not self.noise_stats_enabled:
            self.vx_std = cp.float32(0.0)
            self.vy_std = cp.float32(0.0)
            self.speed_std = cp.float32(0.0)
            self.omega_std = cp.float32(0.0)

    def _weighted_std(self, values, weights):
        """Compute weighted standard deviation."""
        mean = cp.dot(weights, values)
        diff = values - mean
        var = cp.dot(weights, diff * diff)
        return cp.sqrt(var)

    def _update_noise_stats(self):
        """Update noise statistics from current particle set."""
        vx = self.xk[:, 2]
        vy = self.xk[:, 3]
        omega = self.xk[:, 5]
        speed = cp.sqrt(vx * vx + vy * vy)

        w = self.weights
        self.vx_std = self._weighted_std(vx, w)
        self.vy_std = self._weighted_std(vy, w)
        self.speed_std = self._weighted_std(speed, w)
        self.omega_std = self._weighted_std(omega, w)

    def noise_stats(self, host: bool = False):
        """
        Returns std-dev metrics for velocity and angular velocity.

        If `host=True`, converts CuPy scalars to Python floats (GPU->CPU transfer).
        """
        if not self.noise_stats_enabled:
            return None
        print(self.omega_std.get())
        if host:
            return {
                "vx_std": float(self.vx_std.get()),
                "vy_std": float(self.vy_std.get()),
                "speed_std": float(self.speed_std.get()),
                "omega_std": float(self.omega_std.get()),
            }
        return {
            "vx_std": self.vx_std,
            "vy_std": self.vy_std,
            "speed_std": self.speed_std,
            "omega_std": self.omega_std,
        }

    # ------------------------------------------------------------------
    # Internal steps
    # ------------------------------------------------------------------

    def _motion_model(self, dt: float) -> None:
        """Propagate state forward by *dt* seconds."""
        self.xk[:, 0] += self.xk[:, 2] * dt
        self.xk[:, 1] += self.xk[:, 3] * dt
        self.xk[:, 4] += self.xk[:, 5] * dt

        # Process noise (in-place, no allocation)
        self.noise = cp.random.standard_normal(self.noise.shape, dtype=cp.float32)
        self.noise *= self.Q_sqrt_diag
        self.xk += self.noise
        self.xk[:, 5] = self.xk[:, 5] % 31.42 # nyquist thing

    def _measurement_model(self, measurement: cp.ndarray) -> None:
        """Compute predicted measurements for each particle.

        Each particle hypothesises 4 panels at 90-degree intervals; the
        observation is matched to the closest hypothesis.
        """
        x_c = self.xk[:, 0][:, None]  # shape (N,1)
        y_c = self.xk[:, 1][:, None]
        theta = self.xk[:, 4][:, None]
        r = 23.5

        meas = cp.array(measurement, dtype=cp.float32)

        ninety = cp.pi / 2
        ks = cp.arange(4, dtype=cp.float32)  # [0,1,2,3]

        # Precompute sin/cos
        c = cp.cos(theta)
        s = cp.sin(theta)

        # Offsets for 4 hypotheses
        offsets_x = cp.concatenate(
            [r * c, -r * s, -r * c, r * s], axis=1
        )  # shape (N,4)
        offsets_y = cp.concatenate(
            [r * s, r * c, -r * s, -r * c], axis=1
        )  # shape (N,4)

        # Panel positions
        panel_x = x_c + offsets_x
        panel_y = y_c + offsets_y

        # Panel yaws
        panel_yaw = theta + ks[None, :] * ninety  # shape (N,4)

        # Compute squared position error
        dx = panel_x - meas[0]
        dy = panel_y - meas[1]
        pos_dist2 = dx * dx + dy * dy  # shape (N,4)

        # Compute yaw error
        yaw_diff = cp.arctan2(cp.sin(panel_yaw - meas[2]), cp.cos(panel_yaw - meas[2]))
        yaw_dist2 = yaw_diff * yaw_diff  # shape (N,4)

        # Total cost
        cost = pos_dist2 + yaw_dist2  # shape (N,4)

        # Pick best hypothesis per particle
        best_idx = cp.argmin(cost, axis=1)
        arange = cp.arange(self.num_particles)

        self.zk[:, 0] = panel_x[arange, best_idx]
        self.zk[:, 1] = panel_y[arange, best_idx]
        self.zk[:, 2] = panel_yaw[arange, best_idx]

    def _weights(self, measurement: cp.ndarray) -> None:
        """Update particle weights using Mahalanobis distance."""
        diff = self.zk - measurement
        mahal = cp.sum(diff * (diff * self.R_inv_diag), axis=1)

        log_w = -0.5 * mahal
        log_w -= cp.max(log_w)

        w = cp.exp(log_w)
        sum_w = cp.sum(w)
        if sum_w == 0:
            self.weights.fill(1.0 / self.num_particles)
        else:
            self.weights[:] = w / sum_w

    def _resample(self) -> float:
        """Systematic resampling when effective sample size drops."""
        n_eff = 1.0 / cp.dot(self.weights, self.weights)

        if n_eff < self.thresh:
            print("resample")
            N = self.num_particles
            cdf = cp.cumsum(self.weights)
            random_offsets = cp.random.rand(N, dtype=cp.float32)
            positions = (self.indices_base + random_offsets) / N
            indices = cp.searchsorted(cdf, positions, side="left")

            self.resample_buffer[:] = self.xk[indices]
            self.xk, self.resample_buffer = self.resample_buffer, self.xk
            self.weights.fill(1.0 / N)

            self.estimate = cp.mean(self.xk, axis=0)
        else:
            self.estimate = cp.einsum("n,nk->k", self.weights, self.xk)

        if self.noise_stats_enabled:
            self._update_noise_stats()

        return float(n_eff / self.num_particles)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, dt: float, measurement: cp.ndarray):
        """Full predict-update-resample cycle.

        Returns:
            (estimate, confidence) where confidence = N_eff / N.
        """
        self._motion_model(dt)
        self._measurement_model(measurement)
        self._weights(measurement)
        confidence = self._resample()
        return self.estimate, confidence

    def update_with_no_observation(self, dt: float):
        """Predict-only update when no new observation is available."""
        self._motion_model(dt)
        confidence = self._resample()
        return self.estimate, confidence
        
    def reinit(self, prior: cp.ndarray) -> None:
        """Re-initialise all particles around a new prior."""
        self.noise = cp.random.standard_normal(self.noise.shape, dtype=cp.float32)
        self.noise *= self.Q_sqrt_diag
        self.xk[:] = prior + self.noise
        self.weights.fill(1.0 / self.num_particles)
        self.zk.fill(0)
        self.estimate = cp.mean(self.xk, axis=0)

    def prediction(self, dt: float) -> cp.ndarray:
        """Extrapolate the current estimate forward by *dt* seconds."""
        pred = self.estimate.copy()
        pred[0] += pred[2] * dt
        pred[1] += pred[3] * dt
        pred[4] += pred[5] * dt
        return pred

if __name__ == '__main__':
    import time
    prior = cp.array([30, 100, 0.1, 0.1, 40, 0], dtype=cp.float32)
    Q = cp.diag(cp.array([1, 1, 10, 10, 10, 1], dtype=cp.float32))
    R = cp.diag(cp.array([5, 5, 1], dtype=cp.float32))

    pf = ParticleFilter(30_000, Q, R, prior, 3, compute_noise_stats=False)
    measurement = cp.array([100, 100, 30], dtype=cp.float32)
    elapsed = 0

    for x in range(1000):
        start_time = time.perf_counter()
        estimate, confidence = pf.update(elapsed, measurement)
        elapsed = time.perf_counter() - start_time
        stats = pf.noise_stats(host=False)
        print(
            f"\nTime: {elapsed*1000:.2f} ms, Estimate: {estimate}, confidence: {confidence}, "
        )

    pf.reinit(prior)
    print("Reinitialized:", pf.estimate)
