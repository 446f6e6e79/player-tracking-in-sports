"""
Run the 3D reconstruction pipeline end-to-end on three cameras.
The pipeline steps are:
    1. Load each camera's tracking JSON.
    2. Rectify bbox-center points using the camera intrinsics.
    3. Triangulate per-class points across all cameras into a TriangulationOutput.
    4. Persist triangulation.json (always).
    5. Optional renders: top-down minimap MP4, FIFA-style radar overlay on
       camera A's source video, and a 3D matplotlib animation MP4.

Usage examples:
    python scripts/run_3D_reconstruction_pipeline.py --cameras cam_4 cam_13
    python scripts/run_3D_reconstruction_pipeline.py --cameras cam_4 cam_13 \\
        --render-minimap --overlay-video --render-3d-graph
"""
import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import cv2

from src.calibration.camera_data import CameraData
from src.geometry.rectification import rectify_tracking_output
from src.geometry.smoothing import smooth_triangulation
from src.geometry.triangulation import triangulate_rectified_outputs
from src.paths import CameraPaths, ReconstructionPaths, preflight_output_paths
from src.types.tracking import TrackingOutput
from src.utils.video_io import get_frames, open_video
from src.visualization.scene_3d import produce_3d_scene_video
from src.visualization.video_render import (
    produce_minimap_video,
    produce_radar_overlay_video,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run the 3D reconstruction pipeline on three cameras."
    )
    p.add_argument("--cameras", nargs=3, required=True, metavar=("CAM_A", "CAM_B", "CAM_C"),
                   help="Exactly three camera ids, e.g. --cameras cam_4 cam_13 cam_2.")
    p.add_argument("--output-dir", default=None,
                   help="Root directory for results; must match --output-dir used for run_2D_pipeline.py.")
    p.add_argument("--render-minimap", action="store_true",
                   help="Also render a stand-alone top-down minimap MP4.")
    p.add_argument("--overlay-video", action="store_true",
                   help="Also render a radar-overlay MP4 on top of camera A's source video.")
    p.add_argument("--render-3d-graph", action="store_true",
                   help="Also render a 3D matplotlib animation MP4 of the triangulated scene.")
    p.add_argument("--max-frames", type=int, default=-1,
                   help="-1 to read every frame, otherwise stop after N frames "
                        "(only consulted for --overlay-video).")
    p.add_argument("--force", action="store_true",
                   help="Overwrite existing pipeline outputs instead of failing.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cam_a, cam_b, cam_c = args.cameras
    print(f"Running 3D reconstruction pipeline for cameras {cam_a}, {cam_b}, {cam_c}...")

    # Set up paths for each camera and the overall reconstruction outputs
    cam_paths = {
        cam_id: CameraPaths.for_camera(cam_id, results_dir=args.output_dir)
        for cam_id in (cam_a, cam_b, cam_c)
    }
    
    # Ensure all required camera tracking JSONs exist before doing any expensive compute
    recon = ReconstructionPaths.for_cameras(
        (cam_a, cam_b, cam_c),
        results_dir=args.output_dir,
    )
    recon.mkdir()

    output_paths_to_check: list[Path] = [recon.triangulation_json]
    if args.render_minimap:
        output_paths_to_check.append(recon.minimap_video)
    if args.overlay_video:
        output_paths_to_check.append(recon.overlay_video)
    if args.render_3d_graph:
        output_paths_to_check.append(recon.scene_3d_video)

    # Check for existing outputs before doing any expensive compute, to avoid overwriting results
    preflight_output_paths(output_paths_to_check, args.force)

    # 1. Load tracking JSONs and camera calibrations for all cameras
    cameras: dict[str, CameraData] = {}
    trackings: dict[str, TrackingOutput] = {}
    for cam_id in (cam_a, cam_b, cam_c):
        json_path = cam_paths[cam_id].tracking_json
        if not json_path.exists():
            raise FileNotFoundError(f"Tracking JSON not found: {json_path}")
        cameras[cam_id] = CameraData.load(cam_id)
        trackings[cam_id] = TrackingOutput.read(str(json_path))
        print(f"Loaded {cam_id}: {len(trackings[cam_id].frames)} tracking frames")

    # 2. Rectify bbox-center points per camera
    print("Rectifying tracking points...")
    rectified = {
        cam_id: rectify_tracking_output(trackings[cam_id], cameras[cam_id])
        for cam_id in cameras
    }

    # 3. Triangulate across both cameras
    print("Triangulating across cameras...")
    triangulation = triangulate_rectified_outputs(cameras, rectified)
    print(
        f"Triangulated {len(triangulation.frames)} frames across "
        f"{triangulation.camera_ids}"
    )
    print("Smoothing 3D tracks with forward Kalman filter...")
    triangulation = smooth_triangulation(triangulation)

    # 4. Persist the triangulation output (JSON)
    print(f"Writing triangulation output to {recon.triangulation_json}...")
    triangulation.write(str(recon.triangulation_json), overwrite=args.force)

    # 5. Optional renders
    if args.render_minimap:
        print(f"Saving top-down minimap video to {recon.minimap_video}...")
        produce_minimap_video(triangulation, str(recon.minimap_video))

    if args.overlay_video:
        overlay_source = cam_paths[cam_a].video
        if not overlay_source.exists():
            raise FileNotFoundError(f"Video not found: {overlay_source}")
        print(f"Loading {cam_a} frames from {overlay_source} for radar overlay...")
        cap = open_video(str(overlay_source))
        max_frames = None if args.max_frames < 0 else args.max_frames
        frames_color, _ = get_frames(cap, max_frames=max_frames)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        print(f"Loaded {len(frames_color)} frames at {fps:.2f} fps")

        print(f"Saving radar overlay video to {recon.overlay_video}...")
        produce_radar_overlay_video(
            frames_color,
            triangulation,
            str(recon.overlay_video),
        )

    if args.render_3d_graph:
        print(f"Saving 3D scene animation to {recon.scene_3d_video}...")
        produce_3d_scene_video(triangulation, str(recon.scene_3d_video))


if __name__ == "__main__":
    main()
