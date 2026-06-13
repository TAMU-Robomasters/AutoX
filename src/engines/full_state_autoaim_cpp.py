"""Full-state auto-aim engine whose detection runs in C++ (``armor_panel_cpp``).

Same state machine / estimator / ballistics as :class:`FullStateAutoAimEngine`,
but the **C++ detection pipeline owns the camera** (an OpenCV ``VideoCapture``
inside the extension) instead of the shared ffmpeg ``CameraDriver``. So this
engine declares no ``"frames"`` driver -- Python never opens the camera -- and
paces on the cpp detector's own frame sequence via the non-blocking
``detect_latest()``, preserving the ``loop_hz`` predict-between-frames behaviour.

This is the drop-in C++ replacement for the Python ``ClassicalDetectorModule``
vision path. Select it via ``info.yaml`` ``vision_backend: cpp`` (see
``orchestrator.start_engines``); ``vision_backend: python`` keeps the Python
detector + shared camera driver as the fallback.
"""

from typing import cast

from src.core.module import Module
from src.drivers.mcu import McuDriver
from src.engines.full_state_autoaim import FullStateAutoAimEngine
from src.subsystems.vision.cpp_detector.module import CppDetectorModule


class FullStateAutoAimEngineCpp(FullStateAutoAimEngine):
    """Auto-aim engine using the cpp detector, which owns the camera itself."""

    # No "frames" CameraDriver: the cpp detector opens/owns the camera. Keep the
    # MCU driver for the camera->turret transform + firing solution.
    drivers = {"mcu": McuDriver}

    def _build_detection_module(self) -> Module:
        """Use the C++ detector (owns the camera) instead of the Python one."""
        return CppDetectorModule(self.ctx)

    def _ingest_frame(self) -> None:
        """Non-blocking cpp detection; coast between camera frames.

        ``detect_latest()`` runs the pipeline and returns ``True`` only when the
        cpp reader thread has a frame newer than the last we processed (it sets
        ``ctx.panels`` + ``ctx.frame_ts``). When ``False`` no new frame arrived,
        so we coast -- predict-only, or lose the target on timeout -- exactly as
        the base engine does when ``FrameReader.latest()`` yields no new ``seq``.
        """
        detector = cast(CppDetectorModule, self.detection)
        if detector.detect_latest():
            self._process_detected_panels()
        else:
            self.ctx.new_observation = False
            if self.target_timeout.is_expired:
                self._on_target_lost()
