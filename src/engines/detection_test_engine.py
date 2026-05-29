"""Test engine: runs only the classical detector and prints detected panels."""

import time
import numpy as np

from src.core.engine import Engine
from src.subsystems.display import display
from src.subsystems.video_streaming.video_stream import video_stream
from src.subsystems.vision.classical_detector.module import ClassicalDetectorModule
from src.toolbox.globals import config
from src.types.autoaim import ParticleFilterAutoAimContext


class DetectionTestEngine(Engine[ParticleFilterAutoAimContext]):
    """Engine that runs the detector in a loop and prints panel count + FPS."""

    def __init__(self) -> None:
        self.ctx = ParticleFilterAutoAimContext()
        self.detection = ClassicalDetectorModule(self.ctx)
        super().__init__(
            modules=[self.detection],
            context_type=ParticleFilterAutoAimContext,
        )

    def initialize(self) -> None:
        if hasattr(video_stream, "load_threaded_cam"):
            video_stream.load_threaded_cam()
        self._frame_count = 0
        self._last_print = time.perf_counter()

    def execute(self) -> None:
        self.detection.run()
        self._frame_count += 1

    def update(self) -> None:
        if config.log.display_live_frames:
            display.show_windows()

        panels = self.ctx.panels or []
        
        for p in panels:
            print(f"    pos={p.position}  yaw={np.degrees(p.yaw):.3f}", flush=True)

        now = time.perf_counter()
        elapsed = now - self._last_print
        if elapsed >= 1.0:
            fps = self._frame_count / elapsed
            print(f"FPS: {fps:5.1f}  |  panels: {len(panels)}", flush=True)
            self._frame_count = 0
            self._last_print = now


if __name__ == "__main__":
    engine = DetectionTestEngine()
    engine.start()
    try:
        engine.join()
    except KeyboardInterrupt:
        engine.stop()
