"""
Single-camera extrinsics calibration via solvePnP against the 3D court model.

The user clicks court landmarks on the camera's first frame; each landmark has a
known 3D world coordinate (FIBA dimensions, origin at the court center, +Z up —
see `src/geometry/court.py` for more details).

SolvePnP recovers the camera's rvec/tvec directly in the court frame, without the need of an
anchoring step. The user can verify the reprojection residuals before deciding to save the extrinsics to disk.
Usage:
    python scripts/calibrate_extrinsics.py --camera cam_13
"""
import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.calibration.camera_data import CameraData
from src.calibration.extrinsics import solve_camera_pose, verify_stored_extrinsics
from src.calibration.picker import collect_clicks
from src.paths.defaults import DEFAULT_CAMERA_DATA_DIR
from src.cli import add_input_dir_arg
from src.paths import CameraPaths
from src.utils.logging import configure_logging, get_logger
from src.utils.video_io import load_first_frame


logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Single-camera extrinsics calibration via solvePnP.")
    p.add_argument("--camera", required=True,
                   help="Camera id to calibrate (e.g. cam_2, cam_4, cam_13).")
    add_input_dir_arg(p)
    p.add_argument("--display-max-dim", type=int, default=1280,
                   help="Longest side of the click window in pixels.")
    p.add_argument("--verify", action="store_true",
                   help="Report reprojection RMSE for the stored extrinsics without re-clicking "
                        "(requires a prior calibration that saved landmarks_px).")
    return p.parse_args()


def main() -> None:
    configure_logging()
    args = parse_args()

    if args.verify:
        cam_json = DEFAULT_CAMERA_DATA_DIR / f"{args.camera}.json"
        try:
            rmse = verify_stored_extrinsics(cam_json)
            logger.info("Stored extrinsics RMSE for %s: %.2f px", args.camera, rmse)
        except KeyError as e:
            logger.warning("%s", e)
        return

    # Load the camera parameters from disk
    cam = CameraData.load(args.camera)
    paths = CameraPaths.for_camera(args.camera, videos_input_dir=args.input_dir)
    frame = load_first_frame(paths.video)

    print(f"\n=== Click landmarks in {args.camera} ===")
    # Collect 2D-3D correspondences via user clicks
    clicks, status = collect_clicks(args.camera, frame, args.display_max_dim)
    # If the user quit during clicking, we should not write any JSON file
    if status == "quit":
        logger.info("Quit requested. No JSON written.")
        return

    try:
        # Compute the camera pose via solvePnP, and get the reprojection residuals for each landmark
        rvec, tvec, labels, residuals = solve_camera_pose(clicks, cam.mtx, cam.dist)
    except ValueError as e:
        logger.error("solvePnP refused: %s", e)
        return

    print(f"\n=== Reprojection residuals (px), {len(labels)} landmarks ===")
    print(f"mean={residuals.mean():.2f}  max={residuals.max():.2f}")
    for label, residual in zip(labels, residuals):
        print(f"    {label:<32s} residual={residual:.2f}")

    # Prompt the user to confirm writing the extrinsics to disk
    answer = input(f"\nWrite extrinsics for {args.camera}? [y/N]: ").strip().lower()
    if answer != "y":
        logger.info("Discarded. No JSON modified.")
        return
    # Save the extrinsics and the clicked pixels (for future --verify runs)
    clicked_px = {label: clicks[label] for label in labels}
    cam.save_extrinsics(rvec, tvec, landmarks_px=clicked_px)
    logger.info("Wrote rvec/tvec + landmark pixels to %s", cam.path)


if __name__ == "__main__":
    main()
