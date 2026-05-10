import cv2

from src.types.geometry import FrameTriangulatedPoints, TriangulationOutput
from src.types.tracking import TrackingOutput
from src.types.detection import DetectionOutput
from src.utils.video_io import save_video
from src.visualization.drawing import draw_detections, draw_tracked_detections
from src.visualization.minimap import canvas_size, draw_dot, make_base_canvas


_INFO_COLOR = (200, 200, 200)


def produce_detection_output_video(
    frames: list[cv2.Mat],
    detection_output: DetectionOutput,
    output_path: str,
    fps: float | None = None,
    draw_conf: bool = True,
) -> None:
    """Produce an annotated output video from frames and a pre-tracking detection output.
    Parameters:
        - frames: list of original BGR frames (must match len(detection_output.frames))
        - detection_output: per-frame detections (no track_ids)
        - output_path: path for the output MP4
        - fps: frame rate; falls back to detection_output.fps if None
        - draw_conf: whether to overlay confidence scores on boxes
    """
    out_fps = fps if fps is not None else detection_output.fps
    annotated = [
        draw_detections(frame, frame_detections, draw_conf)
        for frame, frame_detections in zip(frames, detection_output.frames)
    ]
    save_video(annotated, output_path, int(out_fps))


def produce_tracking_output_video(
    frames: list[cv2.Mat],
    output: DetectionOutput | TrackingOutput,
    output_path: str,
    fps: float | None = None,
) -> None:
    """Produce an annotated output video from frames and either a detection or
    tracking output — the detection variant simply omits the '#track_id' caption.

    Bounding boxes are colored by team (Ball=yellow, Red=red, White=white,
    Referee=orange). Each box is captioned just above its top edge with
    '{jersey_number} #{track_id} {confidence:.2f}'. Class labels are expected
    to follow the fine-tuned model's schema ('Red_11', 'White_2', 'Refree_1',
    'Ball'); unknown labels fall back to a gray box with the raw class name.

    Parameters:
        - frames: list of original BGR frames (must match len(output.frames))
        - output: per-frame detections — DetectionOutput or TrackingOutput
        - output_path: path for the output MP4
        - fps: frame rate; falls back to output.fps if None
    """
    out_fps = fps if fps is not None else output.fps
    annotated = [
        draw_tracked_detections(frame, frame_detections)
        for frame, frame_detections in zip(frames, output.frames)
    ]
    save_video(annotated, output_path, int(out_fps))


def _render_minimap_frame(
    base_canvas: cv2.Mat,
    triangulated: FrameTriangulatedPoints | None,
) -> cv2.Mat:
    """Build a per-frame minimap by copying the base canvas and stamping any dots."""
    canvas = base_canvas.copy()
    if triangulated is not None:
        for point in triangulated.points:
            draw_dot(canvas, point)
    return canvas


def produce_minimap_video(
    triangulation: TriangulationOutput,
    output_path: str,
    fps: float | None = None,
) -> None:
    """Render the triangulated scene as a stand-alone top-down minimap MP4.

    Each frame draws the FIBA court canvas plus team-coded dots for every
    triangulated `Point3D`, with a `frame {idx}` caption in the upper left.

    Parameters:
        - triangulation: cross-camera triangulated points keyed by frame_index
        - output_path: destination MP4 path
        - fps: frame rate; falls back to triangulation.fps (or 25 if missing)
    """
    out_fps = fps if fps is not None else triangulation.fps
    if out_fps <= 0:
        out_fps = 25.0

    base = make_base_canvas()
    frames: list[cv2.Mat] = []
    for triangulated in triangulation.frames:
        canvas = _render_minimap_frame(base, triangulated)
        cv2.putText(canvas, f"frame {triangulated.frame_index}", (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, _INFO_COLOR, 1, cv2.LINE_AA)
        frames.append(canvas)

    save_video(frames, output_path, int(out_fps))


def _overlay_minimap(
    frame: cv2.Mat,
    minimap: cv2.Mat,
    alpha: float,
) -> cv2.Mat:
    """Alpha-blend `minimap` onto `frame` centered horizontally near the bottom edge.
    Mutates and returns `frame`."""
    h, w = frame.shape[:2]
    mh, mw = minimap.shape[:2]
    x = (w - mw) // 2
    y = h - mh - 10
    roi = frame[y:y + mh, x:x + mw]
    blended = cv2.addWeighted(roi, 1.0 - alpha, minimap, alpha, 0.0)
    frame[y:y + mh, x:x + mw] = blended
    return frame


def produce_radar_overlay_video(
    frames: list[cv2.Mat],
    tracking_output: TrackingOutput,
    triangulation_output: TriangulationOutput,
    output_path: str,
    fps: float | None = None,
    alpha: float = 0.7,
    scale: float = 0.30,
) -> None:
    """Render a FIFA-style annotated video with a top-down minimap radar overlay.

    Each frame gets team-colored bounding boxes (`draw_tracked_detections`) and
    a per-frame triangulated minimap composited centered along the bottom edge.

    Parameters:
        - frames: original BGR frames (must align with tracking_output.frames)
        - tracking_output: 2D bounding-box tracks for the same camera/source
        - triangulation_output: 3D points keyed by frame_index
        - output_path: destination MP4 path
        - fps: frame rate; falls back to tracking_output.fps if None
        - alpha: minimap opacity (0=invisible, 1=opaque). Default 0.7.
        - scale: minimap width as a fraction of source video width. Default 0.30.
    """
    out_fps = fps if fps is not None else tracking_output.fps

    base = make_base_canvas()
    canvas_w, canvas_h = canvas_size()

    # Resize target preserves the canvas aspect ratio.
    video_h, video_w = frames[0].shape[:2]
    target_w = int(video_w * scale)
    target_h = int(canvas_h * target_w / canvas_w)

    # Index triangulation frames by frame_index for O(1) lookup.
    tri_by_index = {f.frame_index: f for f in triangulation_output.frames}
    matched = sum(
        1 for f in tracking_output.frames if f.frame_index in tri_by_index
    )
    print(
        f"Radar overlay: {matched}/{len(tracking_output.frames)} frames "
        f"have matching triangulation data"
    )

    annotated_frames: list[cv2.Mat] = []
    for frame, tracking_frame in zip(frames, tracking_output.frames):
        annotated = draw_tracked_detections(frame, tracking_frame)
        triangulated = tri_by_index.get(tracking_frame.frame_index)
        minimap = _render_minimap_frame(base, triangulated)
        minimap_resized = cv2.resize(
            minimap, (target_w, target_h), interpolation=cv2.INTER_AREA
        )
        annotated = _overlay_minimap(annotated, minimap_resized, alpha)
        annotated_frames.append(annotated)

    save_video(annotated_frames, output_path, int(out_fps))
