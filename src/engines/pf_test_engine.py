"""Test engine: feeds fake panel measurements to the CUDA PF and reports FPS."""

import time
import numpy as np

from src.core.engine import Engine
from src.core.module import Context
from src.subsystems.estimation.full_state.pf import _default_particle_filter


class PFTestEngine(Engine[Context]):
    """Minimal engine that drives the CUDA particle filter with synthetic data.

    No modules — the PF is called directly in execute() to measure raw throughput.
    Prints FPS to stdout once per second.
    """

    def __init__(self) -> None:
        super().__init__(modules=[], context_type=Context)

    def initialize(self) -> None:
        self.pf = _default_particle_filter()

        # Static fake observation: one panel at (50 cm, 100 cm), yaw 0.5 rad
        self.fake_obs = np.array([[50.0, 100.0, 0.5]], dtype=np.float32)
        self.dt = 1.0 / 120.0  # simulate 120 Hz camera

        self._frame_count = 0
        self._last_print = time.perf_counter()
        self._loop_start = time.perf_counter()

    def execute(self) -> None:
        self.pf.update(self.dt, self.fake_obs)
        self._frame_count += 1

    def update(self) -> None:
        now = time.perf_counter()
        elapsed = now - self._last_print
        if elapsed >= 1.0:
            fps = self._frame_count / elapsed
            est = self.pf.estimate
            print(
                f"FPS: {fps:6.1f}  |  "
                f"est: x={est[0]:.1f} y={est[1]:.1f} "
                f"vx={est[2]:.2f} vy={est[3]:.2f} "
                f"θ={est[4]:.3f} ω={est[5]:.3f}",
                flush=True,
            )
            self._frame_count = 0
            self._last_print = now


if __name__ == "__main__":
    engine = PFTestEngine()
    engine.start()
    try:
        engine.join()
    except KeyboardInterrupt:
        engine.stop()
