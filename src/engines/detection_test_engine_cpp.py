"""Test engine: runs the C++ detector and prints detected panels.

Mirror of detection_test_engine.py but driving CppDetectorModule. The camera is
owned by the C++ pipeline (opened in the module's __init__), so unlike the
Python variant this engine does NOT call video_stream.load_threaded_cam().
"""

import time
import numpy as np

from src.core.engine import Engine
from src.subsystems.display import display
from src.subsystems.vision.cpp_detector.module import CppDetectorModule
from src.toolbox.globals import config
from src.types.autoaim import FullStateAutoAimContext


class DetectionTestEngineCpp(Engine[FullStateAutoAimContext]):
    """Engine that runs the C++ detector in a loop and prints panel count + FPS."""

    def __init__(self) -> None:
        self.ctx = FullStateAutoAimContext()
        self.detection = CppDetectorModule(self.ctx)
        super().__init__(
            modules=[self.detection],
            context_type=FullStateAutoAimContext,
        )

    def initialize(self) -> None:
        # C++ owns the camera (opened in CppDetectorModule.__init__); nothing to
        # open here.
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
    engine = DetectionTestEngineCpp()
    engine.start()
    try:
        engine.join()
    except KeyboardInterrupt:
        engine.stop()
