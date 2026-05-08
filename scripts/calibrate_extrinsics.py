"""
Single-camera extrinsics calibration via solvePnP against the 3D court model.

The user clicks court landmarks on the camera's first frame; each landmark has a
known 3D world coordinate (FIBA dimensions, origin at the court center, +Z up —
see `src/geometry/court.py`). solvePnP recovers the camera's rvec/tvec
directly in the court frame — no anchor camera, no relative pose, no separate
scale step.

Usage:
    python scripts/calibrate_extrinsics.py --camera cam_13
"""
import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from src.geometry.camera_data import CameraData
from src.geometry.extrinsics_calibration import solve_camera_pose
from src.geometry.landmark_picker import collect_clicks
from src.utils.video import get_frames, open_video


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Single-camera extrinsics calibration via solvePnP.")
    p.add_argument("--camera", required=True, 
                   help="Camera id to calibrate (e.g. cam_2, cam_4, cam_13).")
    p.add_argument("--input-dir", default="data/videos",
                   help="Directory containing the source videos (outN.mp4).")
    p.add_argument("--display-max-dim", type=int, default=1280,
                   help="Longest side of the click window in pixels.")
    return p.parse_args()


def _load_first_frame(camera_id: str, input_dir: Path) -> np.ndarray:
    """Load the first frame of the given camera's video."""
    path = input_dir / (camera_id.replace("cam_", "out") + ".mp4")
    cap = open_video(str(path))
    try:
        frames, _ = get_frames(cap, max_frames=1)
    finally:
        cap.release()
    if not frames:
        raise RuntimeError(f"Could not read any frame from {path}")
    return frames[0]


def main() -> None:
    args = parse_args()
    cam = CameraData.load(args.camera)
    frame = _load_first_frame(args.camera, Path(args.input_dir))

    print(f"\n=== Click landmarks in {args.camera} ===")
    clicks, status = collect_clicks(args.camera, frame, args.display_max_dim)
    if status == "quit":
        print("Quit requested. No JSON written.")
        return

    try:
        rvec, tvec, labels, residuals = solve_camera_pose(clicks, cam.mtx, cam.dist)
    except ValueError as e:
        print(f"\nsolvePnP refused: {e}")
        return

    print(f"\n=== Reprojection residuals (px), {len(labels)} landmarks ===")
    print(f"mean={residuals.mean():.2f}  max={residuals.max():.2f}")
    for label, residual in zip(labels, residuals):
        print(f"    {label:<32s} residual={residual:.2f}")

    print(f"\ntvec (mm) = [{tvec[0,0]:.1f}, {tvec[1,0]:.1f}, {tvec[2,0]:.1f}]")
    print(f"rvec      = [{rvec[0,0]:.4f}, {rvec[1,0]:.4f}, {rvec[2,0]:.4f}]")

    answer = input(f"\nWrite extrinsics for {args.camera}? [y/N]: ").strip().lower()
    if answer != "y":
        print("Discarded. No JSON modified.")
        return
    cam.save_extrinsics(rvec, tvec)
    print(f"Wrote rvec/tvec to {cam.path}")


if __name__ == "__main__":
    main()
