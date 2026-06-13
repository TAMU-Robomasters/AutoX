"""Test engine: runs only the classical detector and prints detected panels.

Frames come from a shared ``CameraDriver`` over iceoryx2 (the always-shared
model): the engine reads the newest frame via its ``FrameReader`` handle and
writes it to ``ctx.frame`` for the detector module. The module never touches the
camera or IPC.
"""

import time

import numpy as np

from src.core.engine import Engine
from src.drivers.video_stream import CameraDriver
from src.subsystems.display import display
from src.subsystems.vision.classical_detector.module import ClassicalDetectorModule
from src.toolbox.globals import config
from src.types.autoaim import FullStateAutoAimContext


class DetectionTestEngine(Engine[FullStateAutoAimContext]):
    """Engine that runs the detector in a loop and prints panel count + FPS."""

    drivers = {"frames": CameraDriver}

    def __init__(self, driver_registry: dict | None = None) -> None:
        self.ctx = FullStateAutoAimContext()
        self.detection = ClassicalDetectorModule(self.ctx)
        super().__init__(
            modules=[self.detection],
            context_type=FullStateAutoAimContext,
            driver_registry=driver_registry,
        )

    def initialize(self) -> None:
        self.frames = self.driver("frames")  # FrameReader (built in run())
        self._frame_count = 0
        self._last_seq = -1
        self._last_print = time.perf_counter()

    def execute(self) -> None:
        frame = self.frames.latest()
        if frame is None or frame.seq == self._last_seq:
            time.sleep(0.0005)  # no new frame yet; don't busy-spin or reprocess
            return
        self._last_seq = frame.seq
        self.ctx.frame = frame.data
        self.ctx.frame_ts = frame.timestamp
        self.detection.run()
        self._frame_count += 1

    def update(self) -> None:
        if config.log.display_live_frames:
            print("Displaying live frames. This may cause FPS to drop significantly.")
            display.show_windows()

        panels = self.ctx.panels or []

        now = time.perf_counter()
        elapsed = now - self._last_print
        if elapsed >= 1.0:
            fps = self._frame_count / elapsed
            print(f"FPS: {fps:5.1f}  |  panels: {len(panels)}", flush=True)
            self._frame_count = 0
            self._last_print = now


if __name__ == "__main__":
    from src.core.orchestrator import launch_system

    processes = launch_system([DetectionTestEngine])
    try:
        for p in processes:
            p.join()
    except KeyboardInterrupt:
        for p in processes:
            p.terminate()
            p.join()
