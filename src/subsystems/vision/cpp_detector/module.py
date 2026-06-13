"""C++-backed vision module for the auto-aim pipeline.

Drop-in replacement for :class:`ClassicalDetectorModule` that runs the whole
detection chain (frame grab -> threshold/morph/contours -> light pairing ->
panel geometry -> icon classification -> PnP) inside the compiled
``armor_panel_cpp`` extension built from ``cpp/``.

Unlike the pure-Python module, the camera is owned by the C++ side: this module
calls ``video_source_init`` once and then ``detect_panels`` every tick.

Camera intrinsics are read straight from the ``LENSMODEL_OPENCV12``
``.cameramodel`` produced by ``utils/camera_calibration/mrcal_calibrate_camera.py``.
The C++ ``video_source`` hands solvePnP raw (distorted) frames, so the full
OPENCV12 distortion model is the self-consistent choice (mrcal's "normal" mode):
solvePnP undistorts the corners natively, no warp remap needed.
"""
import atexit
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import cv2 as cv
import mrcal
import numpy as np

# module.py -> cpp_detector -> vision -> subsystems -> src -> repo root
_REPO_ROOT = Path(__file__).resolve().parents[4]

# The compiled armor_panel_cpp.so is installed at the repo root. globals.py
# chdir's into src/ on import, so a dynamic cwd entry on sys.path no longer
# points here — pin the absolute repo root so the import works regardless of
# how the process was launched.
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import armor_panel_cpp  # noqa: E402

from src.core.module import Module, real  # noqa: E402
from src.subsystems.display import BLUE, display  # noqa: E402
from src.toolbox.geometry_tools import BoundingBox  # noqa: E402
from src.toolbox.globals import config  # noqa: E402
from src.types.autoaim import ArmorPanel, FullStateAutoAimContext  # noqa: E402

_CAMERAMODEL_PATH = (
    _REPO_ROOT / "assets/intrinsics/mrcal_1280x720/opencv12.cameramodel"
)


def _load_opencv12_intrinsics(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load a 3x3 camera matrix + OpenCV distCoeffs from an OPENCV12 model.

    mrcal's ``LENSMODEL_OPENCV12`` intrinsics vector is
    ``[fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, k5, k6, s1, s2, s3, s4]`` — the
    leading four are the pinhole terms and the trailing twelve are OpenCV's
    ``distCoeffs`` in native order, so ``cv::solvePnP`` consumes them directly.
    """
    model = mrcal.cameramodel(str(path))
    lensmodel, idata = model.intrinsics()
    if lensmodel != "LENSMODEL_OPENCV12":
        raise ValueError(
            f"{path} is {lensmodel}, expected LENSMODEL_OPENCV12 "
            f"(needed so the distortion vector maps onto cv2 distCoeffs)."
        )
    idata = np.asarray(idata, dtype=np.float64)
    fx, fy, cx, cy = idata[:4]
    cam_matrix = np.array(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64
    )
    dist = np.ascontiguousarray(idata[4:], dtype=np.float64)
    return cam_matrix, dist


def _cpp_panels_to_armor_panels(cpp_panels) -> List[ArmorPanel]:
    """Convert C++ ``Panel`` objects into validated :class:`ArmorPanel`."""
    panels: List[ArmorPanel] = []
    for panel in cpp_panels:
        # icon_detection() leaves id == -1 when no icon matched; solve_pnp()
        # skips those, so they carry no usable pose. Mirror the Python module's
        # "skip if icon_detection fails" behavior.
        if panel.id == -1:
            continue

        rvec = np.asarray(panel.rvec, dtype=np.float64)
        # Camera-relative panel yaw: 0 = panel face square to camera, ±90° =
        # edge-on. rvec is a Rodrigues axis*angle vector, so its z-component is
        # not the yaw — extract it from the rotation matrix instead. The PF
        # engine adds turret_yaw on top of this to get global yaw.
        R, _ = cv.Rodrigues(rvec)
        yaw_rad = -np.arctan2(R[0, 2], R[2, 2]) + np.pi
        yaw_rad = (yaw_rad + np.pi) % (2 * np.pi) - np.pi  # wrap to (-π, π]
        if abs(np.degrees(yaw_rad)) >= 90:
            continue

        # panel.corners is the (4, 2) outer panel boundary in
        # [top_left, top_right, bottom_right, bottom_left] order. Match the
        # (N, 1, 2) int32 contour shape the rest of the pipeline expects.
        corners = np.asarray(panel.corners, dtype=np.float32)
        contour = corners.reshape(-1, 1, 2).astype(np.int32)

        if config.log.display_live_frames:
            display.windows["main"].add_contour(contour, color=BLUE)
        panels.append(
            ArmorPanel(
                icon=panel.id,
                position=np.asarray(panel.tvec, dtype=np.float64),
                orientation=rvec,
                bbx=BoundingBox.from_points(
                    top_left=corners[0],
                    bottom_right=corners[2],
                ),
                contour=contour,
                yaw=float(yaw_rad),
            )
        )
    return panels


class CppDetectorModule(Module[FullStateAutoAimContext]):
    """Detects armor panels via the compiled ``armor_panel_cpp`` pipeline."""

    def __init__(self, context: FullStateAutoAimContext):
        super().__init__(
            name="cpp_detector",
            context=context,
            inputs=[],
            outputs=["panels"],
        )

        # process_frame() thresholds the 'r' (red) or 'b'/other (blue) channel;
        # pass our team color's first letter to match frame_proccesing.py.
        self._enemy_color = config.our_team_color[0].lower()

        # Camera + C++ state are set up lazily on the first run() call rather
        # than here. Engines are multiprocessing.Process instances: __init__
        # runs in the parent, but run() runs in the forked child. The C++
        # bufferless reader thread does not survive a fork, so the camera must
        # be opened from inside the child or get_frame() blocks forever.
        self._started = False

        # Non-blocking (detect_latest) bookkeeping: last C++ frame seq we
        # processed, and whether the most recent detect_latest() saw a new frame.
        self._last_seq = -1
        self.frame_is_new = False
        self.frame_seq = -1

    def _ensure_started(self) -> None:
        """Open the camera and push intrinsics into C++ (child-process side)."""
        if self._started:
            return

        # OPENCV12 distortion model + raw frames = mrcal's "normal" mode:
        # solvePnP undistorts the corners itself.
        cam_matrix, dist = _load_opencv12_intrinsics(_CAMERAMODEL_PATH)
        armor_panel_cpp.set_intrinsics(cam_matrix, dist)
        armor_panel_cpp.set_icon_tolerance(float(config.classical.icon_tolerance))

        # Push the light-pairing tunables from info.yaml's `classical:` block so
        # the C++ pairing uses the same values as the Python detector. The
        # height-ratio band is a [lo, hi] list in config.
        c = config.classical
        lo, hi = c.height_ratio_thresh
        armor_panel_cpp.set_pairing_params(
            angle_diff_multiplier=float(c.angle_diff_multiplier),
            misalignment_multiplier=float(c.misalignment_multiplier),
            expected_distance_multiplier=float(c.expected_distance_multiplier),
            height_ratio_multiplier=float(c.height_ratio_multiplier),
            angle_diff_thresh=float(c.angle_diff_thresh),
            misalignment_thresh=float(c.misalignment_thresh),
            height_ratio_thresh_lo=float(lo),
            height_ratio_thresh_hi=float(hi),
            score_thresh=float(c.score_thresh),
        )

        armor_panel_cpp.video_source_init(
            device_id=int(config.hardware.camera_index),
            exposure=float(config.classical.cam_exposure),
        )
        atexit.register(armor_panel_cpp.video_source_close)
        self._started = True

    @real(requires="camera")
    def _run_detect(self) -> Optional[List[ArmorPanel]]:
        """Run the C++ detection pipeline and populate *ctx.panels*."""
        self._ensure_started()

        # Stamp capture time so the engine can compute frame_delay_ms for the
        # camera->turret transform. The C++ pipeline grabs the frame internally
        # and exposes no per-frame timestamp, so we approximate with a
        # perf_counter() reading taken just before detect_panels — same clock
        # the Python detector's Frame.timestamp uses (usb_cam.get_frame).
        self.ctx.frame_ts = time.perf_counter()
        cpp_panels = armor_panel_cpp.detect_panels(self._enemy_color)

        # detect_panels consumed a frame internally; pull a copy for display.
        # get_last_frame() does a full-frame memcpy out of C++, so only pay it
        # when we're actually showing windows.
        if config.log.display_live_frames:
            frame = armor_panel_cpp.get_last_frame()
            if frame is not None:
                display.windows["main"].img = frame

        if not cpp_panels:
            return None

        return _cpp_panels_to_armor_panels(cpp_panels)

    def detect_latest(self) -> bool:
        """Non-blocking detection on the newest frame; the cpp-camera engine path.

        Unlike ``run()`` (which blocks waiting for a frame), this grabs the
        newest frame the C++ reader thread has and only runs the pipeline if its
        sequence number advanced since the last call. This lets the engine loop
        run faster than the camera and predict-only between frames (same role the
        ``FrameReader.latest()`` + ``seq`` dedup plays for the Python detector).

        Side effects: on a new frame, writes ``ctx.panels`` (a list of
        :class:`ArmorPanel`, or ``None``) and stamps ``ctx.frame_ts``.

        Returns:
            ``True`` iff a new frame was processed (``ctx.panels`` updated).
            ``False`` means no new frame arrived -> the engine should coast.
        """
        self._ensure_started()

        # Approximate capture time on the same clock the engine uses for
        # frame_delay_ms. The newest frame is at most ~one camera period old, so
        # reading the clock just before processing is a close (and same-clock)
        # stand-in for the missing per-frame C++ timestamp.
        now = time.perf_counter()
        seq, is_new, cpp_panels = armor_panel_cpp.detect_panels_latest(
            self._enemy_color, self._last_seq
        )
        self.frame_seq = seq
        if not is_new:
            self.frame_is_new = False
            return False

        self._last_seq = seq
        self.frame_is_new = True
        self.ctx.frame_ts = now

        if config.log.display_live_frames:
            frame = armor_panel_cpp.get_last_frame()
            if frame is not None:
                display.windows["main"].img = frame

        self.ctx.panels = (
            _cpp_panels_to_armor_panels(cpp_panels) if cpp_panels else None
        )
        return True
