"""Generate pinhole intrinsics + an undistortion remap from an mrcal cameramodel.

mrcal's splined-stereographic lens model has no native OpenCV support, so we
precompute a per-pixel map that reprojects each raw frame onto an equivalent
pinhole image. Downstream OpenCV code (PnP, etc.) then operates on the pinhole
intrinsics with zero distortion.

Outputs (written into ``--out-dir``, default = cameramodel's parent):
    camera_matrix.pkl  3x3 pinhole K, pickled numpy array
    dist.pkl           zeros, shape (1, 5)
    mapxy.npy          (H_out, W_out, 2) float32 map for mrcal.transform_image

Install mrcal (not on PyPI, system package only):
    sudo apt install python3-mrcal
The uv-managed venv needs system site-packages access to see it
(create with ``uv venv --system-site-packages`` if not already).

Usage:
    python utils/camera_calibration/generate_pinhole_from_mrcal.py
    python utils/camera_calibration/generate_pinhole_from_mrcal.py \
        --cameramodel assets/intrinsics/mrcal_spline_sentry_cam/camera-0.cameramodel
"""
import argparse
import pickle
from pathlib import Path

import mrcal
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CAMERAMODEL = (
    REPO_ROOT / "assets/intrinsics/mrcal_spline_sentry_cam/camera-0.cameramodel"
)


def generate(cameramodel_path: Path, out_dir: Path, fit: str) -> None:
    """Write pinhole intrinsics + remap to ``out_dir``."""
    out_dir.mkdir(parents=True, exist_ok=True)

    model_splined = mrcal.cameramodel(str(cameramodel_path))
    model_pinhole = mrcal.pinhole_model_for_reprojection(model_splined, fit=fit)
    mapxy = mrcal.image_transformation_map(
        model_splined, model_pinhole, intrinsics_only=True
    )

    _, pinhole_intrinsics = model_pinhole.intrinsics()
    fx, fy, cx, cy = pinhole_intrinsics[:4]
    camera_matrix = np.array(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64
    )
    dist = np.zeros((1, 5), dtype=np.float64)

    with (out_dir / "camera_matrix.pkl").open("wb") as f:
        pickle.dump(camera_matrix, f)
    with (out_dir / "dist.pkl").open("wb") as f:
        pickle.dump(dist, f)
    np.save(out_dir / "mapxy.npy", mapxy)

    h, w = mapxy.shape[:2]
    print(f"Wrote pinhole intrinsics ({w}x{h}, fit={fit}) to {out_dir}")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cameramodel",
        type=Path,
        default=DEFAULT_CAMERAMODEL,
        help="Path to the splined mrcal .cameramodel file.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (defaults to the cameramodel's parent directory).",
    )
    parser.add_argument(
        "--fit",
        default="corners",
        help="Reprojection fit mode for mrcal.pinhole_model_for_reprojection.",
    )
    args = parser.parse_args()

    out_dir = args.out_dir or args.cameramodel.parent
    generate(args.cameramodel, out_dir, args.fit)


if __name__ == "__main__":
    main()
