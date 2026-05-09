import functools
from typing import Literal
import cv2
import numpy as np

from src.calibration.extrinsics import LandmarkClicks
from src.geometry.court import LANDMARKS, WORLD_LANDMARKS_MM
from src.visualization.minimap import canvas_size, make_base_canvas, world_to_px

# Header strip stacked above the camera frame; the click callback subtracts
# this to translate window→camera coordinates.
_STATUS_BAR_HEIGHT = 50
_DIAGRAM_INSET_WIDTH = 320
_DIAGRAM_INSET_MARGIN = 20

_GREEN = (0, 255, 0)
_RED = (0, 0, 255)
_BLUE = (255, 0, 0)

# Output type for collect_clicks status: 
# - "done" if user completed all clicks and confirmed 
# - "quit" if they aborted.
PickerStatus = Literal["done", "quit"]


@functools.lru_cache(maxsize=1)
def _base_inset(width: int = _DIAGRAM_INSET_WIDTH) -> np.ndarray:
    """Pre-scaled court canvas, computed once and reused each frame."""
    full = make_base_canvas()
    mw, mh = canvas_size()
    return cv2.resize(full, (width, int(round(mh * width / mw))))


def _inset_world_to_px(x_mm: float, y_mm: float, iw: int, ih: int) -> tuple[int, int]:
    """Scale minimap world_to_px output to inset pixel dimensions."""
    mw, mh = canvas_size()
    mx, my = world_to_px(x_mm, y_mm)
    return int(round(mx * iw / mw)), int(round(my * ih / mh))


def _make_diagram_inset(clicks: LandmarkClicks, next_index: int) -> np.ndarray:
    """Copy the cached court base and overlay landmark state dots."""
    # Get a fresh copy of the base court diagram to draw on
    canvas = _base_inset().copy()

    ih, iw = canvas.shape[:2]
    # For each landmark, determine color and label
    for i, label in enumerate(LANDMARKS):
        # Get the world coordinates of the landmark
        x_mm, y_mm, _ = WORLD_LANDMARKS_MM[label]
        click = clicks.get(label)
        
        # If the landmark has been clicked, show green dot + label
        if click is not None:
            color, label = _GREEN, f"#{i + 1}"
        # If it is the next one to click, show a blue dot and no label
        elif i == next_index:
            color, label = _BLUE, None
        # If it was skipped (user pressed 'n'), show a red dot and no label
        elif i < next_index:
            color, label = _RED, None
        else:
            continue
        # Convert world mm to inset pixels and draw the dot + optional label
        px, py = _inset_world_to_px(x_mm, y_mm, iw, ih)
        cv2.circle(canvas, (px, py), 4, color, -1)
        if label is not None:
            cv2.putText(canvas, label, (px + 5, py - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1, cv2.LINE_AA)
    return canvas


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
    Loop to collect user clicks for each landmark, showing real-time feedback on the camera frame.
    The user clicks the landmarks in order, allowing to undo with 'u' or skip with 'n'. 
    Once all landmarks are clicked, they can press Enter to confirm or 'q' to quit.
    Parameters:
        - camera_id: identifier for the camera (used in the window title and status bar)
        - frame: the camera frame
        - display_max_dim: the maximum dimension for displaying the camera frame
    """
    h, w = frame.shape[:2]
    scale = display_max_dim / max(h, w) if max(h, w) > display_max_dim else 1.0
    if scale != 1.0:
        base_display = cv2.resize(frame, (int(round(w * scale)), int(round(h * scale))))
    else:
        base_display = frame.copy()

    clicks: LandmarkClicks = {label: None for label in LANDMARKS}
    # Stack of clicked/skipped landmarks to support undo
    order: list[str] = []
    index = 0

    def on_mouse(event, x, y, flags, param):
        """Mouse callback to record clicks. Only accepts left button down events, and ignores clicks in the header strip."""
        nonlocal index
        # Skip if not left button 
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if index >= len(LANDMARKS):
            return
        # click landed in the header strip, not the camera frame
        if y < _STATUS_BAR_HEIGHT:
            return  
        # Get the current landmark label to click
        label = LANDMARKS[index]
        # Record the click coordinates, adjusting for the header strip and display scaling
        clicks[label] = (x / scale, (y - _STATUS_BAR_HEIGHT) / scale)
        order.append(label)
        index += 1

    # Create a named window and set the mouse callback
    window = f"calibrate_{camera_id}"
    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window, on_mouse)

    try:
        while True:
            # Render the overlay with the current clicks and instructions, and wait for key input
            cv2.imshow(window, _render_overlay(base_display, camera_id, clicks, index, scale))
            key = cv2.waitKey(20) & 0xFF
            
            # Handle quit case
            if key == ord("q"):
                return clicks, "quit"
            
            # Handle skip case
            if key == ord("n") and index < len(LANDMARKS):
                label = LANDMARKS[index]
                clicks[label] = None
                order.append(label)
                index += 1
            
            # Handle undo case
            if key == ord("u") and order:
                last = order.pop()
                clicks[last] = None
                # Avoid going negative on index if user undoes at the start
                index = max(0, index - 1)
            
            # Handle confirm case
            if key in (13, 10) and index >= len(LANDMARKS):
                return clicks, "done"
    finally:
        cv2.destroyWindow(window)
