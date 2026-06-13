"""ARCHIVED full-state auto-aim engine (pre state-machine refactor).

Kept as the reference implementation of the single-radius pipeline and the only
home of the GPU particle-filter backend (ESTIMATION=PARTICLE_FILTER). The live
engine is ``src/engines/full_state_autoaim.py``.

Full pipeline:
    1. ClassicalDetector  -> panels
    2. RobotClassification -> sentry, hero, standard
    3. (execute logic)     -> decide whether to run targeting
    4. Targeting           -> target_robot
    5. (execute logic)     -> transform panels to turret frame via embedded communicator
    6. FullStateEstimation -> estimate  (particle filter or Kalman filter, see ESTIMATION=*)
    7a. FullStateShotTimingModule     -> solution  (|ω| > omega_spin_threshold)
    7b. FullStateContinuousFireModule -> solution  (|ω| ≤ omega_spin_threshold)
    8. (execute logic)     -> send solution to embedded
"""

import time
from multiprocessing import Queue
from typing import Optional, Union

import cv2 as cv
import numpy as np

from src.core.engine import Engine
from src.drivers.mcu import McuDriver
from src.drivers.video_stream import CameraDriver
from src.subsystems.ballistics.full_state_continuous_fire import (
    FullStateContinuousFireModule,
)
from src.subsystems.ballistics.full_state_shot_timing import FullStateShotTimingModule
from src.subsystems.classification import RobotClassificationModule
from src.subsystems.display import display
from src.subsystems.embedded_communicator import CVState
from src.subsystems.estimation.full_state.kalman_filter import FullStateEstimator
from src.subsystems.estimation.full_state.kf import (
    KalmanFilterEstimationModule,
    _default_full_state_kf,
)
from src.subsystems.estimation.full_state.pf import (
    ParticleFilterEstimationModule,
    _default_particle_filter,
)
from src.subsystems.targeting import TargetingModule
from src.subsystems.vision import ClassicalDepthDetectorModule, ClassicalDetectorModule
from src.subsystems.vision.cpp_detector.module import CppDetectorModule
from src.toolbox.globals import config
from src.toolbox.timeout import Timeout
from src.types.autoaim import BallisticSolution, FullStateAutoAimContext

RESET_TIMEOUT_MS = 500
METERS_TO_CM = 100

EstimationModule = Union[ParticleFilterEstimationModule, KalmanFilterEstimationModule]


#FIXME:  cap.read from the video stream is not blocking until a new frame arrives meaning we are running stuff on the same frame multiple times

class ArchiveFullStateAutoAimEngine(Engine[FullStateAutoAimContext]):
    """Auto-aim engine using a configurable full-state estimator (particle filter or Kalman filter)."""

    drivers = {"frames": CameraDriver, "mcu": McuDriver}

    def __init__(
        self, driver_registry: Optional[dict] = None, queue: Optional[Queue] = None
    ) -> None:
        self._queue = queue
        self.ctx = FullStateAutoAimContext()

        # self.detection = ClassicalDepthDetectorModule(self.ctx)
        # self.detection = ClassicalDetectorModule(self.ctx)
        # C++ detection pipeline (owns the camera; see initialize()).
        # self.detection = CppDetectorModule(self.ctx)
        self.detection = ClassicalDetectorModule(self.ctx)
        self.classification = RobotClassificationModule(self.ctx)
        self.targeting = TargetingModule(self.ctx)

        self.estimation: EstimationModule
        if config.estimation.method == "kalman_filter":
            self.estimation = KalmanFilterEstimationModule(self.ctx)
        else:
            self.estimation = ParticleFilterEstimationModule(self.ctx)

        self.shot_timing = FullStateShotTimingModule(self.ctx)
        self.continuous_fire = FullStateContinuousFireModule(self.ctx)

        super().__init__(
            modules=[
                self.detection,
                self.classification,
                self.targeting,
                self.estimation,
                self.shot_timing,
                self.continuous_fire,
            ],
            context_type=FullStateAutoAimContext,
            driver_registry=driver_registry,
        )

    def initialize(self) -> None:
        """Create child-process resources (MCU client + state estimator)."""
        estimator: FullStateEstimator
        if config.estimation.method == "kalman_filter":
            estimator = _default_full_state_kf()
        else:
            estimator = _default_particle_filter()
        self.estimation.set_estimator(estimator)
        self.shot_timing.set_estimator(estimator)
        self.continuous_fire.set_estimator(estimator)

        self.target_timeout = Timeout(duration=RESET_TIMEOUT_MS / 1E3)  # 500 ms timeout for filter updates

        # MCU access goes through the shared McuDriver (serial/ros/mock chosen by
        # the MCU= profile); the engine no longer opens the UART itself.
        self.mcu = self.driver("mcu")
        # Fallback pitch/yaw sent when no target is available
        self._last_pitch: float = float(np.deg2rad(-10))
        self._last_yaw: float = 0.0
        self.alignment_time_ms: int = 255  # Large default alignment time when no target is present
        self.cv_state: int = CVState.NO_TARGET.value

        # Frames come from the shared CameraDriver via this FrameReader handle
        # (built by the engine base from the driver registry). The detector reads
        # ctx.frame; it never touches the camera or IPC itself.
        self.frames = self.driver("frames")
        self._last_seq = -1


    def execute(self) -> None:
        """Run one iteration of the full-state auto-aim pipeline."""
        self.ctx.start_loop_time = time.perf_counter()

        self.alignment_time_ms = 255
        self.cv_state = CVState.NO_TARGET.value

        # ----------------------------------------------------------
        # 0. Pull the newest frame from the camera driver into the context.
        #    (Fixes the old "reprocessing the same frame" FIXME: FrameReader
        #    hands back the freshest frame each loop.)
        # ----------------------------------------------------------
        frame = self.frames.latest()
        while frame is None or frame.seq == self._last_seq:
            time.sleep(0.0005)  # frame-pace the loop: wait for the next NEW frame
            frame = self.frames.latest()
        self._last_seq = frame.seq
        self.ctx.frame = frame.data
        self.ctx.frame_ts = frame.timestamp

        # ----------------------------------------------------------
        # 1. Detection: frame -> panels
        # ----------------------------------------------------------
        self.detection.run()

        panels = self.ctx.panels
        # print(panels)
        #? why max 3 panels?
        has_panels = panels is not None and 0 < len(panels) < 3


        if not has_panels:
            if self.target_timeout.is_expired:
                self.ctx.target_robot = None
            self.ctx.new_observation = False
        else:
            self.target_timeout.reset()

            # ----------------------------------------------------------
            # Classification: panels -> sentry, hero, standard
            # ----------------------------------------------------------
            self.classification.run()

            # Only update target_robot when classification actually finds one;
            # keep the previous target alive if classification misses a frame
            # so we don't spuriously reinit the PF.
            if self.ctx.standard and self.ctx.standard.panels:
                self.ctx.target_robot = self.ctx.standard

                print(f"before transformation: {self.ctx.target_robot.panels[0].position.flatten()}")
            # ----------------------------------------------------------
            # Transform panels to turret frame via embedded communicator
            # ----------------------------------------------------------
            current_time = time.perf_counter()
            frame_delay_ms: int = int((current_time - self.ctx.frame_ts) * 1000) #- 20
            transformation_data = self.mcu.get_transformation(frame_delay_ms)
            if transformation_data is None:
                # Transformation unavailable — keep predicting without a new observation
                print("warning: no transformation data received from embedded")
                self.ctx.new_observation = False
            else:
                turret_yaw, _turret_pitch, camera_to_turret_matrix = transformation_data
                _transform_panels_to_turret_frame(
                    panels, camera_to_turret_matrix, turret_yaw
                )
                self.ctx.new_observation = True
                if self.ctx.target_robot is not None and self.ctx.target_robot.panels:
                    print(f"Panel position after classification: {self.ctx.target_robot.panels[0].position}")
                    pan = self.ctx.target_robot.panels[0]
                    if self._queue is not None and pan is not None:
                        pass
                        # self._queue.put_nowait(float(np.degrees(pan.yaw)))


        # print(f"Target robot: {self.ctx.target_robot}")

        # print(f"Target panels after transformation: {target.panels[0].position}")
        # ----------------------------------------------------------
        # 5. Estimation: target_robot -> estimate
        # ----------------------------------------------------------
        print("panel angle:", np.degrees(self.ctx.target_robot.panels[0].yaw) if self.ctx.target_robot and self.ctx.target_robot.panels else None)

        self.estimation.run()


        if self.ctx.estimate is None:
            return
        # The rewritten ballistic modules aim at estimate.aim_z (live panel
        # mid-height in the new engine). This archive predates that: keep its
        # legacy fixed config z_offset by filling aim_z here.
        if self.ctx.estimate.aim_z is None:
            self.ctx.estimate.aim_z = float(config.ballistic.z_offset)
        print(f"Translational Velocity: {self.ctx.estimate.value[2]:.2f} cm/s, {self.ctx.estimate.value[3]:.2f} cm/s")
        print("angular velocity:", self.ctx.estimate.value[5])
        print("orientation:", np.degrees(self.ctx.estimate.value[4]))
        print("panel angle:", np.degrees(self.ctx.target_robot.panels[0].yaw) if self.ctx.target_robot and self.ctx.target_robot.panels else None)
        if self._queue is not None:
            self._queue.put_nowait(np.degrees(self.ctx.estimate.value[5]))


        # ----------------------------------------------------------
        # 6. Ballistic solver: estimate -> solution
        #    |ω| > omega_spin_threshold → shot timing (wait for panel alignment)
        #    |ω| ≤ omega_spin_threshold → continuous fire (fire freely)
        # ----------------------------------------------------------
        omega = abs(float(self.ctx.estimate.value[5]))
        if omega > config.ballistic.omega_spin_threshold:
            self.shot_timing.run()
            active_cv_state = CVState.SHOT_TIMING.value
        else:
            self.continuous_fire.run()
            active_cv_state = CVState.CONTINUOUS_FIRE.value

        solution = self.ctx.solution

        if solution is None:
            return

        # ----------------------------------------------------------
        # solution to embedded
        # ----------------------------------------------------------
        self._last_pitch = solution.pitch
        self._last_yaw = solution.yaw + np.deg2rad(config.ballistic.yaw_offset)
        self.alignment_time_ms = solution.alignment_time_ms
        self.cv_state = active_cv_state


    def update(self) -> None:
        """Display windows to screen."""
        print(f"Sending to embedded: pitch={np.rad2deg(self._last_pitch):.2f} deg, yaw={np.rad2deg(self._last_yaw):.2f} deg, alignment_time={self.alignment_time_ms} ms, cv_state={self.cv_state}")
        self.mcu.send_solution(
            pitch=self._last_pitch,
            yaw=self._last_yaw,
            time_until_fire=self.alignment_time_ms,
            cv_state=self.cv_state,
        )
        print("fps:",  1 /(time.perf_counter() - self.ctx.start_loop_time))
        if config.log.display_live_frames:
            display.show_windows()



# ---------------------------------------------------------------------------
# Helpers (not modules -- used inline in execute)
# ---------------------------------------------------------------------------


def _transform_panels_to_turret_frame(
    panels, camera_to_turret_matrix: np.ndarray, turret_yaw: float
) -> None:
    """Transform panel poses from camera frame to turret/ballistic frame.

    Modifies panels in-place. The classical detector already sets ``panel.yaw``
    as the camera-relative panel yaw; here we just add the turret yaw to convert
    it into the global/ballistic frame.
    """
    for panel in panels:
        if panel.position is None or panel.orientation is None:
            continue
        # tvec is in cm -> convert to metres for the 4x4 transform, then back
        pos_m = np.append(panel.position / METERS_TO_CM, 1.0)
        panel.position = (camera_to_turret_matrix @ pos_m)[:3] * METERS_TO_CM

        panel.yaw = panel.yaw + turret_yaw
