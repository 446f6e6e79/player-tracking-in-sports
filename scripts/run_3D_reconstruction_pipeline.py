"""
Run the 3D reconstruction pipeline end-to-end on a pair of cameras.
The pipeline steps are:
    1. Load each camera's tracking JSON.
    2. Rectify bbox-center points using the camera intrinsics.
    3. Triangulate per-class points across both cameras into a TriangulationOutput.
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
from src.geometry.triangulation import triangulate_rectified_outputs
from src.types.tracking import TrackingOutput
from src.utils.video_io import get_frames, open_video
from src.visualization.scene_3d import produce_3d_scene_video
from src.visualization.video_render import (
    produce_minimap_video,
    produce_radar_overlay_video,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run the 3D reconstruction pipeline on a pair of cameras."
    )
    p.add_argument("--cameras", nargs=2, required=True, metavar=("CAM_A", "CAM_B"),
                   help="Exactly two camera ids, e.g. --cameras cam_4 cam_13.")
    p.add_argument("--input-dir", default="data/videos",
                   help="Directory containing the source videos (used by --overlay-video).")
    p.add_argument("--tracking-dir", default="results/tracking",
                   help="Root holding {camera}/serialized/{camera}/tracking.json.")
    p.add_argument("--output-dir", default="results",
                   help="Root directory for produced artifacts.")
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


def _preflight_output_paths(paths: list[Path], force: bool) -> None:
    """Validate all output paths up front so we fail before expensive compute."""
    for path in paths:
        if path.exists() and not force:
            raise FileExistsError(
                f"Refusing to overwrite existing file: {path}. Pass --force to replace it."
            )


def video_path_for(camera: str, input_dir: Path) -> Path:
    """Given a camera id like 'cam_13', return the expected path to its source video within `input_dir`."""
    num = camera.split("_", 1)[1]
    return input_dir / f"out{num}.mp4"


def tracking_json_for(camera: str, tracking_dir: Path) -> Path:
    """Path to a camera's tracking JSON inside the 2D-pipeline output layout."""
    return tracking_dir / camera / "serialized" / camera / "tracking.json"


def main() -> None:
    args = parse_args()
    cam_a, cam_b = args.cameras
    print(f"Running 3D reconstruction pipeline for cameras {cam_a}, {cam_b}...")

    input_dir = Path(args.input_dir)
    tracking_dir = Path(args.tracking_dir)

    # Set up the output directory for this camera pair
    pair_dir = Path(args.output_dir) / "reconstruction" / f"{cam_a}__{cam_b}"
    pair_dir.mkdir(parents=True, exist_ok=True)
    serialized_dir = pair_dir / "serialized"
    serialized_dir.mkdir(parents=True, exist_ok=True)

    # Build paths to script outputs
    triangulation_json_path = serialized_dir / "triangulation.json"
    minimap_path = pair_dir / "minimap.mp4"
    overlay_path = pair_dir / "overlay.mp4"
    scene_3d_path = pair_dir / "scene_3d.mp4"

    output_paths_to_check: list[Path] = [triangulation_json_path]
    if args.render_minimap:
        output_paths_to_check.append(minimap_path)
    if args.overlay_video:
        output_paths_to_check.append(overlay_path)
    if args.render_3d_graph:
        output_paths_to_check.append(scene_3d_path)

    # Check for existing outputs before doing any expensive compute, to avoid overwriting results
    _preflight_output_paths(output_paths_to_check, args.force)

    # 1. Load tracking JSONs and camera calibrations for both cameras
    cameras: dict[str, CameraData] = {}
    trackings: dict[str, TrackingOutput] = {}
    for cam_id in (cam_a, cam_b):
        json_path = tracking_json_for(cam_id, tracking_dir)
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

    # 4. Persist the triangulation output (JSON)
    print(f"Writing triangulation output to {triangulation_json_path}...")
    triangulation.write(str(triangulation_json_path), overwrite=args.force)

    # 5. Optional renders
    if args.render_minimap:
        print(f"Saving top-down minimap video to {minimap_path}...")
        produce_minimap_video(triangulation, str(minimap_path))

    if args.overlay_video:
        video_path = video_path_for(cam_a, input_dir)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        print(f"Loading {cam_a} frames from {video_path} for radar overlay...")
        cap = open_video(str(video_path))
        max_frames = None if args.max_frames < 0 else args.max_frames
        frames_color, _ = get_frames(cap, max_frames=max_frames)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        print(f"Loaded {len(frames_color)} frames at {fps:.2f} fps")

        print(f"Saving radar overlay video to {overlay_path}...")
        produce_radar_overlay_video(
            frames_color,
            triangulation,
            str(overlay_path),
        )

    if args.render_3d_graph:
        print(f"Saving 3D scene animation to {scene_3d_path}...")
        produce_3d_scene_video(triangulation, str(scene_3d_path))


if __name__ == "__main__":
    main()
