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
import numpy as np

from src.geometry import court as cp
from src.geometry.rectification import rectify_tracking_output
from src.geometry.triangulation import triangulate_rectified_outputs
from src.types.tracking import TrackingOutput
from src.utils.drawing import get_team_color_and_number, text_color_for


# Layout: width-fit a 28 m court onto a 1120 px canvas with a 60 px margin
# and 1 m of off-court padding so noise just outside the lines stays visible.
_CANVAS_W = 1120
_CANVAS_H = 600
_MARGIN = 60
_PADDING_MM = 1000.0
_S = (_CANVAS_W - 2 * _MARGIN) / (28000.0 + 2 * _PADDING_MM)  # mm -> px

_BG_COLOR = (30, 30, 30)
_COURT_COLOR = (200, 200, 200)
_HOOP_COLOR = (0, 0, 255)
_INFO_COLOR = (200, 200, 200)
_BALL_RADIUS = 8
_PLAYER_RADIUS = 7


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Render top-down minimap of triangulated detections.")
    p.add_argument("--cameras", nargs="+", default=["cam_4", "cam_13"])
    p.add_argument("--tracking-dir", default="results/tracking",
                   help="Root holding {camera}/serialized/{camera}/tracking.json")
    p.add_argument("--output", default="results/minimap/scene.mp4")
    return p.parse_args()


def world_to_px(x_mm: float, y_mm: float) -> tuple[int, int]:
    """+X toward right hoop, +Y toward stands; flip Y so stands sit at top."""
    px = int(round(_MARGIN + (x_mm + 14000.0 + _PADDING_MM) * _S))
    py = int(round(_CANVAS_H - _MARGIN - (y_mm + 7500.0 + _PADDING_MM) * _S))
    return px, py


def _polyline_to_px(pts_mm: np.ndarray) -> np.ndarray:
    return np.array([world_to_px(float(p[0]), float(p[1])) for p in pts_mm], dtype=np.int32)


def _draw_court(canvas: np.ndarray) -> None:
    open_polys = [
        cp.midline(),
        cp.three_point_polyline(-12425.0),
        cp.three_point_polyline( 12425.0),
    ]
    closed_polys = [
        cp.court_outline(),
        cp.center_circle(),
        cp.lane_left(),
        cp.lane_right(),
    ]
    for pts in open_polys:
        cv2.polylines(canvas, [_polyline_to_px(pts)], False, _COURT_COLOR, 1, cv2.LINE_AA)
    for pts in closed_polys:
        cv2.polylines(canvas, [_polyline_to_px(pts)], True, _COURT_COLOR, 1, cv2.LINE_AA)
    for x_mm in (-12425.0, 12425.0):
        cv2.circle(canvas, world_to_px(x_mm, 0.0), 4, _HOOP_COLOR, -1, cv2.LINE_AA)


def _draw_dot(canvas: np.ndarray, point) -> None:
    color, number = get_team_color_and_number(point.class_name)
    px, py = world_to_px(float(point.x), float(point.y))
    if point.class_name.lower().startswith("ball"):
        cv2.circle(canvas, (px, py), _BALL_RADIUS, color, -1, cv2.LINE_AA)
        return
    cv2.circle(canvas, (px, py), _PLAYER_RADIUS, color, -1, cv2.LINE_AA)
    cv2.circle(canvas, (px, py), _PLAYER_RADIUS, (0, 0, 0), 1, cv2.LINE_AA)
    if number:
        text_color = text_color_for(color)
        (tw, th), _ = cv2.getTextSize(number, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
        cv2.putText(canvas, number, (px - tw // 2, py + th // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, text_color, 1, cv2.LINE_AA)


def _load_rectified(camera_id: str, tracking_dir: Path):
    path = tracking_dir / camera_id / "serialized" / camera_id / "tracking.json"
    tracking = TrackingOutput.read(str(path))
    return rectify_tracking_output(tracking, camera_id), len(tracking.frames)


def main() -> None:
    args = parse_args()
    tracking_dir = Path(args.tracking_dir)

    rectified = {}
    for cam in args.cameras:
        rectified[cam], n_frames = _load_rectified(cam, tracking_dir)
        print(f"Loaded {cam}: {n_frames} frames")

    triangulation = triangulate_rectified_outputs(rectified)
    print(f"Triangulated {len(triangulation.frames)} frames across {triangulation.camera_ids}")

    base = np.full((_CANVAS_H, _CANVAS_W, 3), _BG_COLOR, dtype=np.uint8)
    _draw_court(base)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = triangulation.fps if triangulation.fps > 0 else 25.0
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (_CANVAS_W, _CANVAS_H), True)
    if not writer.isOpened():
        raise RuntimeError(f"Could not open VideoWriter for: {out_path}")

    try:
        for frame in triangulation.frames:
            canvas = base.copy()
            for point in frame.points:
                _draw_dot(canvas, point)
            cv2.putText(canvas, f"frame {frame.frame_index}", (10, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, _INFO_COLOR, 1, cv2.LINE_AA)
            writer.write(canvas)
    finally:
        writer.release()
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
