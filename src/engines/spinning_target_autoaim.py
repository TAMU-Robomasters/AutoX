"""Auto-aim engine for a single, fixed-radius, constant-rate spinning target.

Pipeline (mirrors :mod:`particle_filter_autoaim` but drops classification and
targeting -- there is one target, so whatever panels the detector returns are
*it*):

    1. ClassicalDetector        -> panels
    2. (execute logic)          -> transform panels to turret frame via MCU
    3. EkfEstimation            -> estimate
    4. FullStateContinuousFire  -> solution     (omega is always ≤ threshold)
    5. (execute logic)          -> send solution to embedded
"""

from __future__ import annotations

import time
from multiprocessing import Queue
from typing import Optional

import numpy as np

from src.core.engine import Engine
from src.drivers.mcu import McuDriver
from src.drivers.video_stream import CameraDriver
from src.subsystems.display import display
from src.subsystems.ekf_estimation import (
    EkfEstimationModule,
    _default_ekf_tracker,
)
from src.subsystems.embedded_communicator import CVState
from src.subsystems.full_state_continuous_fire import FullStateContinuousFireModule
from src.subsystems.turret_frame import transform_panels_to_turret_frame
from src.subsystems.vision import ClassicalDetectorModule
from src.toolbox.globals import config
from src.types.autoaim import EkfAutoAimContext


class SpinningTargetAutoAimEngine(Engine[EkfAutoAimContext]):
    """Auto-aim engine that tracks one circular spinning target with an EKF."""

    drivers = {"frames": CameraDriver, "mcu": McuDriver}

    def __init__(
        self,
        driver_registry: Optional[dict] = None,
        queue: Optional[Queue] = None,
    ) -> None:
        self._queue = queue
        self.ctx = EkfAutoAimContext()

        self.detection = ClassicalDetectorModule(self.ctx)
        self.estimation = EkfEstimationModule(self.ctx)
        self.continuous_fire = FullStateContinuousFireModule(self.ctx)

        # FullStateContinuousFireModule was written for the PF context and
        # declares ``inputs=["estimate", "target_robot"]``. ``target_robot``
        # isn't a field on EkfAutoAimContext, but its callback only uses it as
        # a "do we have a target?" sentinel (``if ... target_robot is None:
        # return None``). Remapping to ``panels`` keeps that None-check honest
        # (panels is None when nothing is detected) and lets _validate_wiring
        # see all inputs as produced.
        self.continuous_fire.remap_inputs(
            "full_state_continuous_fire",
            ["target_robot"],
            ["panels"],
        )

        super().__init__(
            modules=[
                self.detection,
                self.estimation,
                self.continuous_fire,
            ],
            context_type=EkfAutoAimContext,
            driver_registry=driver_registry,
        )

    # ------------------------------------------------------------------
    # Child-process resource creation
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        """Build the EKF, attach it to estimation + ballistics, open drivers."""
        self.tracker = _default_ekf_tracker()
        self.estimation.set_estimator(self.tracker)
        # FullStateContinuousFireModule duck-types its ``_pf`` -- it just calls
        # ``prediction(dt)`` on it, which EkfTracker exposes.
        self.continuous_fire.set_particle_filter(self.tracker)

        self.mcu = self.driver("mcu")
        self.frames = self.driver("frames")
        self._last_seq = -1

        # Fallbacks while we don't yet have a solution.
        self._last_pitch: float = float(np.deg2rad(-10))
        self._last_yaw: float = 0.0
        self.alignment_time_ms: int = 255
        self.cv_state: int = CVState.NO_TARGET.value

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def execute(self) -> None:
        self.ctx.start_loop_time = time.perf_counter()
        self.alignment_time_ms = 255
        self.cv_state = CVState.NO_TARGET.value

        # 0. Pull the newest frame from the camera driver.
        frame = self.frames.latest()
        while frame is None or frame.seq == self._last_seq:
            time.sleep(0.0005)
            frame = self.frames.latest()
        self._last_seq = frame.seq
        self.ctx.frame = frame.data
        self.ctx.frame_ts = frame.timestamp

        # 1. Detection: frame -> panels.
        self.detection.run()

        # 2. Camera-frame -> turret-frame transform.
        panels = self.ctx.panels
        has_panels = panels is not None and len(panels) > 0
        if has_panels:
            current_time = time.perf_counter()
            frame_delay_ms = int((current_time - self.ctx.frame_ts) * 1000)
            transformation_data = self.mcu.get_transformation(frame_delay_ms)
            if transformation_data is None:
                print("warning: no transformation data received from embedded")
                self.ctx.new_observation = False
            else:
                turret_yaw, _turret_pitch, camera_to_turret_matrix = transformation_data

                transform_camera_to_turret_frame(
                    panels, camera_to_turret_matrix, turret_yaw
                )
                self.ctx.new_observation = True
        else:
            self.ctx.new_observation = False

        # 3. EKF estimation -- run even with no observation so we keep
        # extrapolating between detection misses.
        self.estimation.run()
        if self.ctx.estimate is None:
            return

        # 4. Ballistics -- always continuous-fire for this target (omega is
        # known and below the spin threshold).
        self.continuous_fire.run()
        solution = self.ctx.solution
        if solution is None:
            return

        # 5. Stash for ``update()`` to push to embedded.
        self._last_pitch = solution.pitch
        self._last_yaw = solution.yaw + np.deg2rad(config.ballistic.yaw_offset)
        self.alignment_time_ms = solution.alignment_time_ms
        self.cv_state = CVState.CONTINUOUS_FIRE.value

    def update(self) -> None:
        print(
            f"Sending to embedded: pitch={np.rad2deg(self._last_pitch):.2f} deg, "
            f"yaw={np.rad2deg(self._last_yaw):.2f} deg, "
            f"alignment_time={self.alignment_time_ms} ms, "
            f"cv_state={self.cv_state}"
        )
        self.mcu.send_solution(
            pitch=self._last_pitch,
            yaw=self._last_yaw,
            time_until_fire=self.alignment_time_ms,
            cv_state=self.cv_state,
        )
        if self.ctx.start_loop_time is not None:
            print(f"fps: {1 / (time.perf_counter() - self.ctx.start_loop_time):.1f}")
        if config.log.display_live_frames:
            display.show_windows()


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
