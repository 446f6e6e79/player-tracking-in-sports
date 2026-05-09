"""
Top-down minimap canvas for the FIBA court — reusable rendering surface
shared by `scripts/render_minimap.py` and any future consumer that needs to
project triangulated 3D points onto a 2D layout.
"""
import cv2
import numpy as np

from src.geometry import court as cp
from src.types.geometry import Point3D
from src.utils.drawing import get_team_color_and_number, text_color_for


# Layout: width-fit a 28 m court onto a 1120 px canvas with a 60 px margin
# and 1 m of off-court padding so noise just outside the lines stays visible.
_CANVAS_W = 1120
_CANVAS_H = 600
_MARGIN = 60
_PADDING_MM = 1000.0
_MM_TO_PX = (_CANVAS_W - 2 * _MARGIN) / (28000.0 + 2 * _PADDING_MM)

_BG_COLOR = (30, 30, 30)
_COURT_COLOR = (200, 200, 200)
_HOOP_COLOR = (0, 0, 255)
_BALL_RADIUS = 8
_PLAYER_RADIUS = 7


def canvas_size() -> tuple[int, int]:
    """(width, height) of the minimap canvas in pixels — what VideoWriter wants."""
    return _CANVAS_W, _CANVAS_H


def world_to_px(x_mm: float, y_mm: float) -> tuple[int, int]:
    """+X toward right hoop, +Y toward stands; flip Y so stands sit at top."""
    px = int(round(_MARGIN + (x_mm + 14000.0 + _PADDING_MM) * _MM_TO_PX))
    py = int(round(_CANVAS_H - _MARGIN - (y_mm + 7500.0 + _PADDING_MM) * _MM_TO_PX))
    return px, py


def _polyline_to_px(pts_mm: np.ndarray) -> np.ndarray:
    return np.array([world_to_px(float(p[0]), float(p[1])) for p in pts_mm], dtype=np.int32)


def draw_court(canvas: np.ndarray) -> None:
    """Stroke court lines and hoop centers onto an existing canvas in place."""
    for line in cp.court_lines():
        cv2.polylines(canvas, [_polyline_to_px(line.pts)], line.closed, _COURT_COLOR, 1, cv2.LINE_AA)
    for x_mm in (cp.LEFT_RIM_X, cp.RIGHT_RIM_X):
        cv2.circle(canvas, world_to_px(x_mm, 0.0), 4, _HOOP_COLOR, -1, cv2.LINE_AA)


def make_base_canvas() -> np.ndarray:
    """Allocate a fresh canvas and stroke the court lines on it."""
    canvas = np.full((_CANVAS_H, _CANVAS_W, 3), _BG_COLOR, dtype=np.uint8)
    draw_court(canvas)
    return canvas


def draw_landmark_dots(
    canvas: np.ndarray,
    dots: list[tuple[float, float, tuple[int, int, int], str | None]],
) -> None:
    """Draw colored dots at world (x_mm, y_mm) positions onto a minimap canvas.

    Each entry in `dots` is (x_mm, y_mm, bgr_color, label_or_None).
    """
    for x_mm, y_mm, color, label in dots:
        px, py = world_to_px(x_mm, y_mm)
        cv2.circle(canvas, (px, py), 6, color, -1)
        if label is not None:
            cv2.putText(canvas, label, (px + 8, py - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)


def draw_dot(canvas: np.ndarray, point: Point3D) -> None:
    """Draw a single triangulated point: ball as a yellow dot, players as
    team-colored dots with a black outline and the jersey number centered."""
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
