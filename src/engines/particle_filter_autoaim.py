"""Particle-filter auto-aim engine.

Full pipeline:
    1. ClassicalDetector  -> panels
    2. RobotClassification -> sentry, hero, standard
    3. (execute logic)     -> decide whether to run targeting
    4. Targeting           -> target_robot
    5. (execute logic)     -> transform panels to turret frame via embedded communicator
    6. ParticleFilterEstimation -> estimate
    7. BallisticSolver     -> solution
    8. (execute logic)     -> send solution to embedded
"""

import cv2 as cv
import numpy as np

from src.core.engine import Engine
from src.subsystems.ballistic_solver import BallisticSolverModule
from src.subsystems.classification import RobotClassificationModule
from src.subsystems.display import display
from src.subsystems.embedded_communicator import EmbeddedCommunicator
from src.subsystems.estimation import (
    ParticleFilterEstimationModule,
    _default_particle_filter,
)
from src.subsystems.targeting import TargetingModule
from src.subsystems.vision import ClassicalDetectorModule
from src.types.autoaim import ParticleFilterAutoAimContext
from src.toolbox.globals import config
from src.subsystems.video_streaming.video_stream import create_video_stream

METERS_TO_CM = 100


class ParticleFilterAutoAimEngine(Engine[ParticleFilterAutoAimContext]):
    """Auto-aim engine using a particle filter for state estimation."""

    def __init__(self) -> None:
        self.ctx = ParticleFilterAutoAimContext()

        # Modules
        self.detection = ClassicalDetectorModule(self.ctx)
        # Remap outputs to match ParticleFilterAutoAimContext field names
        # ClassicalDetectorModule outputs "panels" which matches our context

        self.classification = RobotClassificationModule(self.ctx)
        self.targeting = TargetingModule(self.ctx)
        self.estimation = ParticleFilterEstimationModule(self.ctx)
        self.ballistic = BallisticSolverModule(self.ctx)

        super().__init__(
            modules=[
                self.detection,
                self.classification,
                self.targeting,
                self.estimation,
                self.ballistic,
            ],
            context_type=ParticleFilterAutoAimContext,
        )

    def initialize(self) -> None:
        """Create child-process resources (communicator + particle filter)."""
        pf = _default_particle_filter()
        self.estimation.set_particle_filter(pf)
        self.ballistic.set_particle_filter(pf)

        self.communicator = EmbeddedCommunicator()
        # Fallback pitch/yaw sent when no target is available
        self._last_pitch: float = float(np.deg2rad(-10))
        self._last_yaw: float = 0.0

        # create video stream instance
        self.video_stream = create_video_stream()

    def execute(self) -> None:
        """Run one iteration of the particle-filter auto-aim pipeline."""
        self.ctx.frame = self.video_stream.get_frame()
        display.windows["main"].img = self.ctx.frame
        # ----------------------------------------------------------
        # 1. Detection: frame -> panels
        # ----------------------------------------------------------
        self.detection.run()

        panels = self.ctx.panels
        #? why max 3 panels?
        has_panels = panels is not None and 0 < len(panels) < 3

        if not has_panels:
            # No panels: send last known angles with large alignment time,
            # reset the filter, and return early.
            print("No panels detected")
            self.communicator.send_angles_to_embedded(
                pitch=self._last_pitch,
                yaw=self._last_yaw,
                time_until_next_fire=10_000,
                cv_state=0,
            )
            self.estimation.reset() #! should not reset unless timeout
            self.ctx.target_robot = None
            return
        # ----------------------------------------------------------
        # 2. Classification: panels -> sentry, hero, standard
        # ----------------------------------------------------------
        self.classification.run()

        # ----------------------------------------------------------
        # 3. Targeting (only if we haven't selected a robot yet)
        # ----------------------------------------------------------
        if self.ctx.target_robot is None or not self.ctx.target_robot.panels:
            self.targeting.run()
        
        print(f"Target robot: {self.ctx.target_robot}")

        target = self.ctx.target_robot
        if target is None or not target.panels:
            # All classified robots are empty somehow
            self.communicator.send_angles_to_embedded(
                pitch=self._last_pitch,
                yaw=self._last_yaw,
                time_until_next_fire=10_000,
                cv_state=0,
            )
            self.estimation.reset()
            return

        # ----------------------------------------------------------
        # 4. Transform panels to turret frame via embedded communicator
        # ----------------------------------------------------------
        frame_delay_ms: int = 0
        transformation_data = self.communicator.get_camera_to_ballistic_transformation(
            frame_delay_ms
        )
        if transformation_data is None:
            # Could not get transformation; skip this frame
            print("warning: no transformation data received from embedded")
            return

        turret_yaw, _turret_pitch, camera_to_turret_matrix = transformation_data
        _transform_panels_to_turret_frame(
            target.panels, camera_to_turret_matrix, turret_yaw
        )

        # ----------------------------------------------------------
        # 5. Estimation: target_robot -> estimate
        # ----------------------------------------------------------
        self.estimation.run()

        if self.ctx.estimate is None:
            return

        # ----------------------------------------------------------
        # 6. Ballistic solver: estimate -> solution
        # ----------------------------------------------------------
        self.ballistic.run()

        solution = self.ctx.solution
        if solution is None:
            return

        # ----------------------------------------------------------
        # 7. Send solution to embedded
        # ----------------------------------------------------------
        self._last_pitch = solution.pitch
        self._last_yaw = solution.yaw

        print(f"Sending angles to embedded: pitch={solution.pitch:.3f}, yaw={solution.yaw:.3f}, alignment_time={solution.alignment_time_ms}ms")
        self.communicator.send_angles_to_embedded(
            pitch=solution.pitch,
            yaw=solution.yaw,
            time_until_next_fire=solution.alignment_time_ms,
            cv_state=1,
        )

    def update(self) -> None:
        """Display windows to screen."""
        if config.log.display_live_frames:
            display.show_windows()
        

# ---------------------------------------------------------------------------
# Helpers (not modules -- used inline in execute)
# ---------------------------------------------------------------------------


def _transform_panels_to_turret_frame(
    panels, camera_to_turret_matrix: np.ndarray, turret_yaw: float
) -> None:
    """Transform panel positions from camera frame to turret/ballistic frame.

    Modifies panels in-place.
    """
    for panel in panels:
        if panel.position is None or panel.orientation is None:
            continue
        # tvec is in cm -> convert to metres for the 4x4 transform, then back
        pos_m = np.append(panel.position / METERS_TO_CM, 1.0)
        panel.position = (camera_to_turret_matrix @ pos_m)[:3] * METERS_TO_CM

        R, _ = cv.Rodrigues(panel.orientation)
        panel.yaw = (-np.arctan2(R[0, 2], R[2, 2]) + np.deg2rad(180)) + turret_yaw
