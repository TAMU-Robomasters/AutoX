"""Generate pinhole intrinsics + an undistortion remap from an mrcal cameramodel.

mrcal's splined-stereographic lens model has no native OpenCV support, so we
precompute a per-pixel map that reprojects each raw frame onto an equivalent
pinhole image. Downstream OpenCV code (PnP, etc.) then operates on the pinhole
intrinsics with zero distortion.

mrcal.pinhole_model_for_reprojection always sets the destination pinhole's
principal point to the geometric center of the output image; the ``--fit`` flag
only controls how the focal length / output dims are chosen:

    centers-horizontal  (default) — output dims close to source dims; minor
                                    vertical cropping. Use when you want the
                                    transformed frame to look similar in size
                                    to the raw camera frame.
    centers-vertical               — symmetric option; minor horizontal
                                    cropping.
    corners                        — maximize FOV; output dims grow to contain
                                    all four source corners. Larger frame.

Each run writes to its own subdirectory:
    <cameramodel_parent>/pinhole_<fit_tag>_<WxH>/
        camera_matrix.pkl  3x3 pinhole K, pickled numpy array
        dist.pkl           zeros, shape (1, 5)
        mapxy.npy          (H_out, W_out, 2) float32 map for mrcal.transform_image

That layout means trying a different camera model or fit mode never overwrites
a previous run. Point info.yaml's hardware.camera_intrinsics_path at the
specific subdir to use it.

After the artifacts are written the script offers an optional pinhole-recal
sanity check: apply the remap to a set of calibration images, run mrgingham +
mrcal-calibrate-cameras on the transformed images as a LENSMODEL_PINHOLE solve,
and compare the recalibrated K against the script-computed K. They should
match closely if the splined calibration is internally consistent. The
resulting .cameramodel is kept under ``utils/camera_calibration/pinhole_recalibrations/``
for inspection; it is not used by the runtime pipeline.

Install mrcal + mrgingham (not on PyPI, system packages only):
    sudo apt install python3-mrcal mrgingham
The uv-managed venv needs system site-packages access to see mrcal
(create with ``uv venv --python /usr/bin/python3.10 --system-site-packages``).

Usage:
    python utils/camera_calibration/generate_pinhole_from_mrcal.py
    python utils/camera_calibration/generate_pinhole_from_mrcal.py --fit corners
    python utils/camera_calibration/generate_pinhole_from_mrcal.py \
        --cameramodel assets/intrinsics/some_other_cam/camera-0.cameramodel
"""
import argparse
import pickle
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import cv2 as cv
import mrcal
import numpy as np
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[2]
INTRINSICS_ROOT = REPO_ROOT / "assets/intrinsics"
DEFAULT_CAMERAMODEL = INTRINSICS_ROOT / "mrcal_spline_sentry_cam/camera-0.cameramodel"
SCRIPT_DIR = Path(__file__).resolve().parent
RECAL_DIR = SCRIPT_DIR / "pinhole_recalibrations"

_FIT_TAG = {
    "corners": "corners",
    "centers-horizontal": "centersH",
    "centers-vertical": "centersV",
}


def generate(cameramodel_path: Path, fit: str, force: bool) -> Path:
    """Build the pinhole model + remap and write outputs to a per-variant subdir.

    Returns the output directory path.
    """
    model_splined = mrcal.cameramodel(str(cameramodel_path))
    model_pinhole = mrcal.pinhole_model_for_reprojection(model_splined, fit=fit)
    mapxy = mrcal.image_transformation_map(
        model_splined, model_pinhole, intrinsics_only=True
    )

    h, w = mapxy.shape[:2]
    out_dir = cameramodel_path.parent / f"pinhole_{_FIT_TAG[fit]}_{w}x{h}"
    if out_dir.exists() and not force:
        raise SystemExit(
            f"Output directory already exists: {out_dir}\n"
            f"Re-run with --force to overwrite, or pick a different --fit."
        )
    out_dir.mkdir(parents=True, exist_ok=True)

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
    # Also write the pinhole model as a .cameramodel so projection-diff can
    # compare it against the recalibrated pinhole (mrcal CLI needs a file).
    model_pinhole.write(str(out_dir / "pinhole.cameramodel"))

    print(
        f"Wrote pinhole intrinsics ({w}x{h}, fit={fit}) to {out_dir}\n"
        f"  fx={fx:.3f}  fy={fy:.3f}  cx={cx:.3f}  cy={cy:.3f}  "
        f"(image center: {(w - 1) / 2:.1f}, {(h - 1) / 2:.1f})"
    )
    return out_dir


def _prompt(message: str, default: Optional[str] = None) -> str:
    """Prompt with an optional default shown in brackets."""
    suffix = f" [{default}]" if default else ""
    value = input(f"{message}{suffix}: ").strip()
    return value or (default or "")


def offer_uncertainty_check(
    cameramodel_path: Path, fit: str, out_dir: Path
) -> None:
    """Optionally recalibrate a pinhole model from transformed images and compare.

    Walks the user through:
        1. transforming a set of calibration images through the freshly built
           remap, into a temp directory
        2. running mrgingham on those transformed images to detect corners
        3. running mrcal-calibrate-cameras --lensmodel LENSMODEL_PINHOLE
        4. comparing the recalibrated K to the script-computed K
        5. opening mrcal-show-projection-uncertainty on the new model

    The recalibrated .cameramodel is kept; the transformed images are deleted.
    """
    answer = input(
        "\nRecalibrate a pinhole model from transformed images "
        "and view its projection uncertainty? [y/N] "
    ).strip().lower()
    if answer not in ("y", "yes"):
        return

    if shutil.which("mrgingham") is None:
        print("mrgingham not found on PATH. Install with: sudo apt install mrgingham")
        return
    if shutil.which("mrcal-calibrate-cameras") is None:
        print("mrcal-calibrate-cameras not found on PATH (apt install python3-mrcal).")
        return
    if shutil.which("mrcal-show-projection-uncertainty") is None:
        print(
            "mrcal-show-projection-uncertainty not found on PATH "
            "(apt install python3-mrcal)."
        )
        return
    if shutil.which("mrcal-show-projection-diff") is None:
        print(
            "mrcal-show-projection-diff not found on PATH "
            "(apt install python3-mrcal)."
        )
        return

    image_dir_str = _prompt("Calibration images directory")
    if not image_dir_str:
        print("No directory provided. Skipping.")
        return
    image_dir = Path(image_dir_str).expanduser().resolve()
    if not image_dir.is_dir():
        print(f"Not a directory: {image_dir}. Skipping.")
        return

    image_glob = _prompt("Image glob", "*.jpg")
    images = sorted(image_dir.glob(image_glob))
    if not images:
        print(f"No images matched {image_dir}/{image_glob}. Skipping.")
        return
    print(f"Found {len(images)} images.")

    try:
        object_spacing = float(_prompt("Chessboard object spacing (meters)"))
        object_width_n = int(_prompt("Chessboard width (corners across)"))
    except ValueError:
        print("Could not parse number. Aborting.")
        return

    mapxy = np.load(out_dir / "mapxy.npy")
    with (out_dir / "camera_matrix.pkl").open("rb") as f:
        K_script = pickle.load(f)
    H_out, W_out = int(mapxy.shape[0]), int(mapxy.shape[1])
    focal_init = int(round((float(K_script[0, 0]) + float(K_script[1, 1])) / 2))

    RECAL_DIR.mkdir(parents=True, exist_ok=True)
    output_name = (
        f"{cameramodel_path.parent.name}_pinhole_"
        f"{_FIT_TAG[fit]}_{W_out}x{H_out}.cameramodel"
    )
    output_cameramodel = RECAL_DIR / output_name

    with tempfile.TemporaryDirectory(prefix="mrcal_xform_") as tmp:
        tmp_dir = Path(tmp)
        images_dir = tmp_dir / "images"
        images_dir.mkdir()

        print(f"\nTransforming {len(images)} images into {images_dir} ...")
        transformed_images = []
        for img_path in tqdm(images, desc="transform", unit="img"):
            raw = cv.imread(str(img_path))
            if raw is None:
                tqdm.write(f"  skip (unreadable): {img_path.name}")
                continue
            xfrm = mrcal.transform_image(raw, mapxy)
            out_path = images_dir / img_path.name
            cv.imwrite(str(out_path), xfrm)
            transformed_images.append(out_path)
        if not transformed_images:
            print("No images transformed. Aborting.")
            return

        corners_path = tmp_dir / "corners.vnl"
        print(
            f"Detecting corners with mrgingham on {len(transformed_images)} "
            f"images (no per-image progress; this can take a couple of minutes) ..."
        )
        mrgingham_cmd = [
            "mrgingham",
            "--gridn", str(object_width_n),
            *[str(p) for p in transformed_images],
        ]
        with corners_path.open("wb") as f:
            r = subprocess.run(mrgingham_cmd, stdout=f, stderr=subprocess.PIPE)
        if r.returncode != 0:
            print(f"mrgingham failed:\n{r.stderr.decode(errors='replace')}")
            return

        print(
            "Running mrcal-calibrate-cameras with LENSMODEL_PINHOLE "
            "(solver runs to convergence; no progress signal) ..."
        )
        # mrcal-calibrate-cameras takes one glob per camera, not per file.
        images_glob = str(images_dir / "*")
        calib_cmd = [
            "mrcal-calibrate-cameras",
            "--corners-cache", str(corners_path),
            "--lensmodel", "LENSMODEL_PINHOLE",
            "--focal", str(focal_init),
            "--object-spacing", str(object_spacing),
            "--object-width-n", str(object_width_n),
            "--outdir", str(tmp_dir),
            images_glob,
        ]
        r = subprocess.run(calib_cmd, capture_output=True, text=True)
        if r.returncode != 0:
            print("mrcal-calibrate-cameras failed:")
            print(r.stdout)
            print(r.stderr)
            return
        # surface mrcal's own residual summary to the operator
        if r.stdout:
            print(r.stdout)

        produced = sorted(tmp_dir.glob("camera-*.cameramodel"))
        if not produced:
            print("Calibration finished but no .cameramodel was produced.")
            return
        shutil.copy(produced[0], output_cameramodel)

    print(f"\nRecalibrated pinhole model:\n  {output_cameramodel}")

    m_recal = mrcal.cameramodel(str(output_cameramodel))
    _, p_recal = m_recal.intrinsics()
    fx_r, fy_r, cx_r, cy_r = (float(x) for x in p_recal[:4])
    fx_s, fy_s = float(K_script[0, 0]), float(K_script[1, 1])
    cx_s, cy_s = float(K_script[0, 2]), float(K_script[1, 2])

    print("\nPinhole intrinsics comparison "
          "(script = pinhole_model_for_reprojection; recal = LENSMODEL_PINHOLE solve):")
    header = f"  {'param':<6}{'script':>16}{'recal':>16}{'diff':>14}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for label, s, r_ in (("fx", fx_s, fx_r), ("fy", fy_s, fy_r),
                        ("cx", cx_s, cx_r), ("cy", cy_s, cy_r)):
        print(f"  {label:<6}{s:>16.4f}{r_:>16.4f}{r_ - s:>+14.4f}")

    print("\nLaunching mrcal-show-projection-uncertainty ...")
    subprocess.run(["mrcal-show-projection-uncertainty", str(output_cameramodel)])

    print(
        "\nLaunching mrcal-show-projection-diff "
        "(script-computed pinhole vs. recalibrated pinhole) ..."
    )
    subprocess.run([
        "mrcal-show-projection-diff",
        str(out_dir / "pinhole.cameramodel"),
        str(output_cameramodel),
    ])


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
        "--fit",
        default="centers-horizontal",
        choices=sorted(_FIT_TAG.keys()),
        help="Pinhole focal-length fit mode (see module docstring).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing output subdirectory.",
    )
    args = parser.parse_args()

    cameramodel_path = args.cameramodel.resolve()
    if not cameramodel_path.is_file():
        raise SystemExit(
            f"--cameramodel must point to a .cameramodel file, got: {cameramodel_path}"
        )

    out_dir = generate(cameramodel_path, args.fit, args.force)

    if out_dir.is_relative_to(INTRINSICS_ROOT):
        relative = out_dir.relative_to(INTRINSICS_ROOT)
        print(
            f"\nTo use this in info.yaml, set:\n"
            f'    camera_intrinsics_path: "{relative.as_posix()}"'
        )
    else:
        print(
            f"\nOutput is outside {INTRINSICS_ROOT}; copy or move it under "
            f"assets/intrinsics/ and update camera_intrinsics_path accordingly."
        )

    offer_uncertainty_check(cameramodel_path, args.fit, out_dir)


if __name__ == "__main__":
    main()
