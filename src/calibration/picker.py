"""
Interactive OpenCV landmark picker used by the extrinsics-calibration flow.

The user is walked through `LANDMARKS` in order: each iteration shows the
camera frame as canvas, a court-diagram inset highlighting the next target
in blue (already-clicked = green with ordinal, skipped = red), and a status
strip stacked above the frame so the header never obscures clicks.
"""

import functools
import time
from pathlib import Path
from typing import Literal

import cv2
import numpy as np

from src.calibration.extrinsics import LandmarkClicks
from src.geometry.court import LANDMARKS

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DIAGRAM_PATH = _REPO_ROOT / "media" / "court-diagram.drawio.png"

# Header strip stacked above the camera frame; the click callback subtracts
# this to translate window→camera coordinates.
_STATUS_BAR_HEIGHT = 50
# Drop EVENT_LBUTTONDOWN events arriving faster than this (macOS Cocoa fires duplicates).
_CLICK_DEBOUNCE_S = 0.25
_DIAGRAM_INSET_WIDTH = 320
_DIAGRAM_INSET_MARGIN = 20

_GREEN = (0, 255, 0)
_RED = (0, 0, 255)
_BLUE = (255, 0, 0)
_GREEN_RGBA = (0, 255, 0, 255)
_RED_RGBA = (0, 0, 255, 255)
_BLUE_RGBA = (255, 0, 0, 255)

PickerStatus = Literal["done", "quit"]


# Normalized [0, 1] coordinates on the court-diagram PNG, used to render the
# next-target highlight in the click picker. Values are arbitrary visual hints —
# the **label** is what tells the user which physical point to click.
_LANDMARK_DIAGRAM_NORM: dict[str, tuple[float, float]] = {
    # Court corners (4)
    "corner_left_bench":   (0.020, 0.900),
    "corner_right_bench":  (0.980, 0.900),
    "corner_left_stands":  (0.020, 0.150),
    "corner_right_stands": (0.980, 0.150),

    # Center line (4)
    "center_circle_bench":   (0.500, 0.615),
    "center_circle_stands":  (0.500, 0.435),
    "center_line_bench":     (0.500, 0.900),
    "center_line_stands":    (0.500, 0.150),

    # Free-throw lane anchors (8)
    "lane_left_endline_bench":   (0.020, 0.615),
    "lane_left_ft_bench":        (0.205, 0.615),
    "lane_left_endline_stands":  (0.020, 0.455),
    "lane_left_ft_stands":       (0.205, 0.455),
    "lane_right_endline_bench":  (0.980, 0.615),
    "lane_right_ft_bench":       (0.790, 0.615),
    "lane_right_endline_stands": (0.980, 0.455),
    "lane_right_ft_stands":      (0.790, 0.455),

    # Hoops + floor-below anchors (4)
    "hoop_left_rim":          (0.065, 0.520),
    "floor_under_hoop_left":  (0.065, 0.520),
    "hoop_right_rim":         (0.930, 0.520),
    "floor_under_hoop_right": (0.930, 0.520),

    # Three-point arcs (6)
    "three_pt_left_endline_bench":   (0.020, 0.840),
    "three_pt_left_apex":            (0.300, 0.510),
    "three_pt_left_endline_stands":  (0.020, 0.220),
    "three_pt_right_endline_bench":  (0.980, 0.840),
    "three_pt_right_apex":           (0.700, 0.510),
    "three_pt_right_endline_stands": (0.980, 0.220),

    # Backboard outer corners (8). Top vs. bottom is invisible in this top-down
    # inset, so the (norm_x, norm_y) offsets are just disambiguation hints —
    # the label tells the user which corner to click.
    "backboard_left_top_bench":      (0.055, 0.555),
    "backboard_left_top_stands":     (0.055, 0.485),
    "backboard_left_bottom_bench":   (0.075, 0.555),
    "backboard_left_bottom_stands":  (0.075, 0.485),
    "backboard_right_top_bench":     (0.945, 0.555),
    "backboard_right_top_stands":    (0.945, 0.485),
    "backboard_right_bottom_bench":  (0.925, 0.555),
    "backboard_right_bottom_stands": (0.925, 0.485),
}

# Drift guard: every world landmark must have a diagram coord, and vice-versa.
assert set(_LANDMARK_DIAGRAM_NORM) == set(LANDMARKS), (
    "_LANDMARK_DIAGRAM_NORM keys must match LANDMARKS exactly. "
    f"Missing: {set(LANDMARKS) - set(_LANDMARK_DIAGRAM_NORM)}, "
    f"extra: {set(_LANDMARK_DIAGRAM_NORM) - set(LANDMARKS)}"
)


@functools.lru_cache(maxsize=1)
def _load_diagram_inset(width: int = _DIAGRAM_INSET_WIDTH) -> np.ndarray:
    """Load the transparent court-diagram PNG, resized to `width` (RGBA preserved)."""
    if not _DIAGRAM_PATH.exists():
        raise FileNotFoundError(f"Court diagram image not found: {_DIAGRAM_PATH}")
    diagram = cv2.imread(str(_DIAGRAM_PATH), cv2.IMREAD_UNCHANGED)
    if diagram is None:
        raise RuntimeError(f"OpenCV failed to decode {_DIAGRAM_PATH}")
    if diagram.ndim != 3 or diagram.shape[2] != 4:
        raise RuntimeError(f"Expected an RGBA PNG, got shape {diagram.shape}")
    h0, w0 = diagram.shape[:2]
    scale = width / w0
    return cv2.resize(diagram, (width, max(1, int(round(h0 * scale)))))


def _alpha_composite(canvas: np.ndarray, rgba: np.ndarray, top_left: tuple[int, int]) -> None:
    """Alpha-blend an RGBA image onto a BGR canvas in place."""
    x0, y0 = top_left
    h, w = rgba.shape[:2]
    roi = canvas[y0:y0 + h, x0:x0 + w].astype(np.float32)
    alpha = rgba[..., 3:4].astype(np.float32) / 255.0
    rgb = rgba[..., :3].astype(np.float32)
    canvas[y0:y0 + h, x0:x0 + w] = (alpha * rgb + (1.0 - alpha) * roi).astype(np.uint8)


def _diagram_pixel(label: str, inset_w: int, inset_h: int) -> tuple[int, int] | None:
    """Pixel coordinates of `label` on the inset, or None for unknown labels."""
    norm = _LANDMARK_DIAGRAM_NORM.get(label)
    if norm is None:
        return None
    nx, ny = norm
    return int(round(nx * inset_w)), int(round(ny * inset_h))


def _draw_diagram_dot(
    inset: np.ndarray,
    label: str,
    color: tuple[int, int, int, int],
    radius: int = 4,
    ordinal: int | None = None,
) -> None:
    """Place a colored dot on the diagram inset for `label`, with an optional ordinal label."""
    ih, iw = inset.shape[:2]
    target = _diagram_pixel(label, iw, ih)
    if target is None:
        return
    cv2.circle(inset, target, radius, color, thickness=-1)
    if ordinal is not None:
        cv2.putText(inset, f"#{ordinal}",
                    (target[0] + radius + 2, target[1] - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)


def _draw_cam_dot(cam: np.ndarray, xy: tuple[int, int], ordinal: int) -> None:
    """Place a green dot + ordinal on the camera frame at clicked pixel `xy`."""
    cv2.circle(cam, xy, 4, _GREEN, thickness=-1)
    cv2.putText(cam, f"#{ordinal}", (xy[0] + 6, xy[1] - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, _GREEN, 1, cv2.LINE_AA)


def _render_status_bar(width: int, dtype: np.dtype, camera_id: str, next_index: int) -> np.ndarray:
    """Two-line header strip stacked above the camera frame."""
    if next_index < len(LANDMARKS):
        msg = f"{camera_id}  [{next_index + 1}/{len(LANDMARKS)}] click {LANDMARKS[next_index]}"
    else:
        msg = f"{camera_id}  all landmarks recorded — Enter to continue"
    header = np.zeros((_STATUS_BAR_HEIGHT, width, 3), dtype=dtype)
    cv2.putText(header, msg, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(header, "keys: u=undo  n=not visible  q=quit",
                (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
    return header


def _render_overlay(
    camera_frame_resized: np.ndarray,
    camera_id: str,
    clicks: LandmarkClicks,
    next_index: int,
    scale: float,
) -> np.ndarray:
    """Compose camera frame (with click markers), diagram inset progress map, and status bar."""
    cam = camera_frame_resized.copy()
    diagram_inset = _load_diagram_inset().copy()
    ih, iw = diagram_inset.shape[:2]
    inset_top_left = (
        max(0, cam.shape[1] - iw - _DIAGRAM_INSET_MARGIN),
        _STATUS_BAR_HEIGHT + _DIAGRAM_INSET_MARGIN,
    )

    for i, label in enumerate(LANDMARKS):
        click = clicks.get(label)
        if click is not None:
            ix, iy = click
            _draw_cam_dot(cam, (int(round(ix * scale)), int(round(iy * scale))), ordinal=i + 1)
            _draw_diagram_dot(diagram_inset, label, _GREEN_RGBA, ordinal=i + 1)
        elif i == next_index:
            _draw_diagram_dot(diagram_inset, label, _BLUE_RGBA)
        elif i < next_index:
            _draw_diagram_dot(diagram_inset, label, _RED_RGBA)

    _alpha_composite(cam, diagram_inset, inset_top_left)
    return np.vstack([_render_status_bar(cam.shape[1], cam.dtype, camera_id, next_index), cam])


def collect_clicks(
    camera_id: str,
    frame: np.ndarray,
    display_max_dim: int,
) -> tuple[LandmarkClicks, PickerStatus]:
    """
    Walk the user through LANDMARKS, collecting one click per visible landmark.
    Keys: `n` to skip the current landmark, `u` to undo the last entry, Enter
    to confirm once all landmarks have been recorded, `q` to abort.

    Returns (clicks, status) where status is "done" or "quit".
    """
    h, w = frame.shape[:2]
    scale = display_max_dim / max(h, w) if max(h, w) > display_max_dim else 1.0
    if scale != 1.0:
        base_display = cv2.resize(frame, (int(round(w * scale)), int(round(h * scale))))
    else:
        base_display = frame.copy()

    clicks: LandmarkClicks = {label: None for label in LANDMARKS}
    order: list[str] = []  # labels in click order, for undo
    index = 0
    last_click_time = 0.0

    def on_mouse(event, x, y, flags, param):
        nonlocal index, last_click_time
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if index >= len(LANDMARKS):
            return
        if y < _STATUS_BAR_HEIGHT:
            return  # click landed in the header strip, not the camera frame
        now = time.monotonic()
        if now - last_click_time < _CLICK_DEBOUNCE_S:
            return
        last_click_time = now
        label = LANDMARKS[index]
        clicks[label] = (x / scale, (y - _STATUS_BAR_HEIGHT) / scale)
        order.append(label)
        index += 1

    window = f"calibrate_{camera_id}"
    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window, on_mouse)

    try:
        while True:
            cv2.imshow(window, _render_overlay(base_display, camera_id, clicks, index, scale))
            key = cv2.waitKey(20) & 0xFF
            if key == ord("q"):
                return clicks, "quit"
            if key == ord("n") and index < len(LANDMARKS):
                label = LANDMARKS[index]
                clicks[label] = None
                order.append(label)
                index += 1
            if key == ord("u") and order:
                last = order.pop()
                clicks[last] = None
                index = max(0, index - 1)
            if key in (13, 10) and index >= len(LANDMARKS):
                return clicks, "done"
    finally:
        cv2.destroyWindow(window)
