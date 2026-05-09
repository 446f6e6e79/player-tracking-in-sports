import time
from typing import Literal

import cv2
import numpy as np

from src.calibration.extrinsics import LandmarkClicks
from src.geometry.court import LANDMARKS, WORLD_LANDMARKS_MM
from src.visualization.minimap import make_base_canvas, world_to_px

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

PickerStatus = Literal["done", "quit"]


def _make_diagram_inset(clicks: LandmarkClicks, next_index: int) -> np.ndarray:
    """Render a procedural court minimap with landmark state dots, resized to inset width."""
    canvas = make_base_canvas()
    for i, label in enumerate(LANDMARKS):
        x_mm, y_mm, _ = WORLD_LANDMARKS_MM[label]
        px, py = world_to_px(x_mm, y_mm)
        click = clicks.get(label)
        if click is not None:
            cv2.circle(canvas, (px, py), 6, _GREEN, -1)
            cv2.putText(canvas, f"#{i + 1}", (px + 8, py - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, _GREEN, 1, cv2.LINE_AA)
        elif i == next_index:
            cv2.circle(canvas, (px, py), 6, _BLUE, -1)
        elif i < next_index:
            cv2.circle(canvas, (px, py), 6, _RED, -1)
    h, w = canvas.shape[:2]
    scale = _DIAGRAM_INSET_WIDTH / w
    return cv2.resize(canvas, (_DIAGRAM_INSET_WIDTH, int(round(h * scale))))


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

    diagram_inset = _make_diagram_inset(clicks, next_index)
    ih, iw = diagram_inset.shape[:2]
    x0 = max(0, cam.shape[1] - iw - _DIAGRAM_INSET_MARGIN)
    y0 = _STATUS_BAR_HEIGHT + _DIAGRAM_INSET_MARGIN
    cam[y0:y0 + ih, x0:x0 + iw] = diagram_inset

    for i, label in enumerate(LANDMARKS):
        click = clicks.get(label)
        if click is not None:
            ix, iy = click
            _draw_cam_dot(cam, (int(round(ix * scale)), int(round(iy * scale))), ordinal=i + 1)

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
