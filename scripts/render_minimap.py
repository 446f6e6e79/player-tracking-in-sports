"""
Reconstruct the 3D scene as a top-down minimap MP4.

Pipeline: per-camera tracking JSON -> rectified 2D points -> triangulated 3D
points (world frame, mm) -> projected onto a top-down 28x15 m court canvas
frame-by-frame.

    python scripts/render_minimap.py
    python scripts/render_minimap.py --cameras cam_2 cam_4 cam_13
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
from src.visualization.minimap import canvas_size, draw_dot, make_base_canvas


_INFO_COLOR = (200, 200, 200)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Render top-down minimap of triangulated detections.")
    p.add_argument("--cameras", nargs="+", default=["cam_4", "cam_13"])
    p.add_argument("--tracking-dir", default="results/tracking",
                   help="Root holding {camera}/serialized/{camera}/tracking.json")
    p.add_argument("--output", default="results/minimap/scene.mp4")
    return p.parse_args()


def _load_rectified(camera: CameraData, tracking_dir: Path):
    path = tracking_dir / camera.camera_id / "serialized" / camera.camera_id / "tracking.json"
    tracking = TrackingOutput.read(str(path))
    return rectify_tracking_output(tracking, camera), len(tracking.frames)


def main() -> None:
    args = parse_args()
    tracking_dir = Path(args.tracking_dir)

    cameras = {cam_id: CameraData.load(cam_id) for cam_id in args.cameras}

    rectified = {}
    for cam_id, camera in cameras.items():
        rectified[cam_id], n_frames = _load_rectified(camera, tracking_dir)
        print(f"Loaded {cam_id}: {n_frames} frames")

    triangulation = triangulate_rectified_outputs(cameras, rectified)
    print(f"Triangulated {len(triangulation.frames)} frames across {triangulation.camera_ids}")

    base = make_base_canvas()
    width, height = canvas_size()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = triangulation.fps if triangulation.fps > 0 else 25.0
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height), True)
    if not writer.isOpened():
        raise RuntimeError(f"Could not open VideoWriter for: {out_path}")

    try:
        for frame in triangulation.frames:
            canvas = base.copy()
            for point in frame.points:
                draw_dot(canvas, point)
            cv2.putText(canvas, f"frame {frame.frame_index}", (10, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, _INFO_COLOR, 1, cv2.LINE_AA)
            writer.write(canvas)
    finally:
        writer.release()
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
