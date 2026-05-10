import cv2

from src.types.tracking import FrameDetections, FrameTrackedDetections


def text_color_for(bg_bgr: tuple[int, int, int]) -> tuple[int, int, int]:
    """Pick black or white text for legibility against a colored background."""
    b, g, r = bg_bgr
    # Use the ITU-R BT.601 formula to compute perceived luminance from the BGR color.
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    # If the background is bright, use black text; otherwise, use white text.
    return (0, 0, 0) if luminance > 128 else (255, 255, 255)


def _draw_caption(
    frame: cv2.Mat,
    x: int,
    y: int,
    text: str,
    bg_color: tuple[int, int, int],
) -> None:
    """Render a filled-background caption strip just above (x, y) on `frame`.
    Mutates `frame` in place. Text color is auto-picked for contrast."""
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(frame, (x, y - th - 6), (x + tw + 4, y), bg_color, -1)
    cv2.putText(frame, text, (x + 2, y - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color_for(bg_color), 1, cv2.LINE_AA)


def draw_detections(
    frame: cv2.Mat,
    frame_detections: FrameDetections,
    draw_conf: bool = True,
) -> cv2.Mat:
    """Given a frame and its corresponding FrameDetections, draw the bounding boxes and confidence scores on the frame.
    Parameters:
        - frame: BGR image to annotate (not modified in place)
        - frame_detections: FrameDetections containing detections to draw
        - draw_conf: whether to overlay confidence scores
    Returns:
        Annotated copy of the input frame.
    """
    # Start with a copy of the original frame to draw on
    annotated = frame.copy()
    bbox_color = (0, 255, 0)  # green for all pure detections

    # For each detection in the frame, draw its bounding box and optionally its confidence score
    for detection in frame_detections.detections:
        x1, y1, x2, y2 = detection.get_int_bbox_tuple()
        cv2.rectangle(annotated, (x1, y1), (x2, y2), bbox_color, 2)

        # If draw_conf is True, overlay the confidence score as text near the bounding box
        if draw_conf:
            _draw_caption(annotated, x1, y1, f"{detection.confidence:.2f}", bbox_color)

    return annotated


def draw_tracked_detections(
    frame: cv2.Mat,
    frame_detections: FrameDetections | FrameTrackedDetections,
) -> cv2.Mat:
    """Draw team-colored bounding boxes with a '{number} #{track_id} {conf}'
    caption rendered just above each box.

    Color groups: Ball=yellow, Red team=red, White team=white, Referee=orange.
    The number is parsed from the class label ('Red_11' -> '11'); Ball has no
    number, so its caption falls back to 'Ball'. When the input is a plain
    FrameDetections (no track ids on the detections) the '#id' segment is omitted.
    """
    annotated = frame.copy()
    for detection in frame_detections.detections:
        x1, y1, x2, y2 = detection.get_int_bbox_tuple()
        bbox_color = detection.color
        number = detection.number

        # Draw the bounding box in the team color
        cv2.rectangle(annotated, (x1, y1), (x2, y2), bbox_color, 2)

        primary = number if number else detection.class_name
        track_id = getattr(detection, "track_id", None)
        track_part = f" #{track_id}" if track_id is not None else ""
        caption = f"{primary}{track_part} {detection.confidence:.2f}"

        _draw_caption(annotated, x1, y1, caption, bbox_color)

    return annotated
