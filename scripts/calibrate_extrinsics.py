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
from src.cli import add_input_dir_arg
from src.config import get_config
from src.paths import CameraPaths
from src.paths.defaults import DEFAULT_CAMERA_DATA_DIR
from src.utils.logging import configure_logging, get_logger
from src.utils.video_io import load_first_frame


logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Single-camera extrinsics calibration via solvePnP.")
    p.add_argument("--camera", required=True,
                   help="Camera id to calibrate (e.g. cam_2, cam_4, cam_13).")
    add_input_dir_arg(p)
    p.add_argument("--display-max-dim", type=int, default=None,
                   help="Longest side of the click window in pixels (default: from config).")
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
    cfg = get_config()
    display_max_dim = args.display_max_dim if args.display_max_dim is not None else cfg.calibration.display_max_dim

    cam = CameraData.load(args.camera)
    paths = CameraPaths.for_camera(args.camera, videos_input_dir=args.input_dir)
    frame = load_first_frame(paths.video)

    print(f"\n=== Click landmarks in {args.camera} ===")
    clicks, status = collect_clicks(args.camera, frame, display_max_dim)
    if status == "quit":
        logger.info("Quit requested. No JSON written.")
        return

    try:
        rvec, tvec, labels, residuals = solve_camera_pose(
            clicks, cam.mtx, cam.dist,
            min_points=cfg.calibration.solve_pnp_min_points,
        )
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
