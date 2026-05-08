"""
Two-camera calibration of camera extrinsics from cross-camera correspondences.
The user clicks the same set of physical landmarks in both cameras (no world
knowledge required, just "click the same thing in each view").

Pipeline:
    1. The anchor camera (default cam_13) defines the world frame: R=I, T=0.
    2. User clicks LANDMARKS in both anchor and the other camera
       (`src.utils.landmark_picker.collect_clicks`).
    3. Essential matrix + recoverPose -> relative pose of the second camera
       (||t||=1). Triangulate the clicked landmarks in the anchor's frame
       (`src.geometry.extrinsics_calibration.triangulate_landmark_pair`).
    4. Set absolute scale by demanding |rim_center - floor_under_rim| = 3050 mm
       (`src.geometry.extrinsics_calibration.recover_metric_scale`).
    5. Per-camera reprojection diagnostics, then a single y/N prompt before
       atomically writing both JSONs via CameraData.save_extrinsics.

Usage:
    python scripts/calibrate_extrinsics.py                       # cam_13 + cam_4
    python scripts/calibrate_extrinsics.py --camera cam_2        # cam_13 + cam_2
    python scripts/calibrate_extrinsics.py --anchor cam_4 --camera cam_2
"""
import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from src.geometry.extrinsics_calibration import (
    build_extrinsics,
    recover_metric_scale,
    reprojection_residuals,
    triangulate_landmark_pair,
)
from src.geometry.camera_data import CameraData
from src.utils.landmark_picker import collect_clicks
from src.utils.video import get_frames, open_video


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Two-camera extrinsics calibration via cross-camera correspondences.")
    p.add_argument("--anchor", default="cam_13",
                   help="Camera defining the world frame (R=I, t=0). Default: cam_13.")
    p.add_argument("--camera", default="cam_4",
                   help="The other camera to calibrate against the anchor. Default: cam_4.")
    p.add_argument("--input-dir", default="data/videos",
                   help="Directory containing the source videos (outN.mp4).")
    p.add_argument("--display-max-dim", type=int, default=1280,
                   help="Longest side of the display window in pixels.")
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
    if args.anchor == args.camera:
        print(f"--anchor and --camera must be different (both are {args.anchor}).")
        return

    anchor, cam_b = args.anchor, args.camera
    cameras = [anchor, cam_b]
    input_dir = Path(args.input_dir)

    cams: dict[str, CameraData] = {c_id: CameraData.load(c_id) for c_id in cameras}
    frames: dict[str, np.ndarray] = {c_id: _load_first_frame(c_id, input_dir) for c_id in cameras}

    all_clicks: dict[str, dict] = {}
    for c_id in cameras:
        print(f"\n=== Click landmarks in {c_id} ===")
        clicks, status = collect_clicks(c_id, frames[c_id], args.display_max_dim)
        if status == "quit":
            print("Quit requested. No JSONs written.")
            return
        all_clicks[c_id] = clicks

    R_b, t_b, structure, n_inliers, n_pairs = triangulate_landmark_pair(
        all_clicks[anchor], all_clicks[cam_b],
        cams[anchor].mtx, cams[anchor].dist,
        cams[cam_b].mtx, cams[cam_b].dist,
    )
    print(f"\n{anchor} <-> {cam_b}: {n_inliers}/{n_pairs} inliers from recoverPose")
    if n_pairs < 8:
        print(f"Need >=8 common landmarks between {anchor} and {cam_b}, got {n_pairs}.")
        return
    if n_inliers < 6:
        print(f"Too few inliers ({n_inliers}). Re-click with cleaner landmarks.")
        return

    scale_result = recover_metric_scale(structure)
    if scale_result is None:
        print("No scale pair was triangulated — cannot fix metric scale.")
        return
    scale, details = scale_result
    print(f"\n=== Scale anchor: averaged over {len(details)} pair(s) -> scale = {scale:.4f} ===")
    for pair, measured, pair_scale in details:
        print(f"    {pair.a:<28s} <-> {pair.b:<28s} "
              f"measured={measured:.4f}  target={pair.distance_mm:.0f}mm  scale={pair_scale:.4f}")

    t_b_scaled = (t_b * scale).astype(np.float32)
    structure_scaled = {l: (p * scale).astype(np.float32) for l, p in structure.items()}
    extrinsics = build_extrinsics(anchor, cam_b, R_b, t_b_scaled)

    print("\n=== Reprojection residuals (px) ===")
    for cid in cameras:
        rvec, tvec = extrinsics[cid]
        labels, residuals, _ = reprojection_residuals(
            structure_scaled, all_clicks[cid], rvec, tvec, cams[cid].mtx, cams[cid].dist,
        )
        if not labels:
            print(f"{cid}: no labels with both 3D structure and clicks — skipping diagnostic")
            continue
        print(f"{cid}: mean={residuals.mean():.2f}  max={residuals.max():.2f}  n={len(labels)}")
        for label, residual in zip(labels, residuals):
            print(f"    {label:<24s} residual={residual:.2f}")

    answer = input("\nWrite extrinsics for both cameras? [y/N]: ").strip().lower()
    if answer != "y":
        print("Discarded. No JSONs modified.")
        return
    for cid, (rvec, tvec) in extrinsics.items():
        cams[cid].save_extrinsics(rvec, tvec)
        print(f"Wrote rvec/tvec to {cams[cid].path}")


if __name__ == "__main__":
    main()
