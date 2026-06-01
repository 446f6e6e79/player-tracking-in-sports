import itertools
import os
from collections.abc import Iterable

import cv2

from src.types.geometry import FrameTriangulatedPoints, TriangulationOutput
from src.types.tracking import FrameTrackedDetections, TrackingOutput
from src.types.detection import DetectionOutput, FrameDetections
from src.utils.logging import get_logger
from src.utils.video_io import iter_video_frames, save_video
from src.visualization.drawing import draw_detections, draw_tracked_detections, overlay_inset
from src.visualization.minimap import _MARGIN, draw_dot, make_base_canvas


logger = get_logger(__name__)


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


def stream_tracking_output_video(
    annotated_frames: Iterable[cv2.Mat],
    output_path: str,
    fps: float,
) -> None:
    """Write `annotated_frames` to `output_path` one frame at a time.

    Mirrors `save_video` but consumes an iterable, so callers can render
    frames as they are produced (e.g. while streaming chunks from disk) and
    never hold the full annotated video in memory.
    """
    iterator = iter(annotated_frames)
    try:
        first = next(iterator)
    except StopIteration:
        raise ValueError("stream_tracking_output_video received no frames.")

    height, width = first.shape[:2]
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, int(fps), (width, height), True)
    if not writer.isOpened():
        raise RuntimeError(f"Could not open VideoWriter for: {output_path}")
    try:
        writer.write(cv2.cvtColor(first, cv2.COLOR_GRAY2BGR) if first.ndim == 2 else first)
        for frame in iterator:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR) if frame.ndim == 2 else frame)
    finally:
        writer.release()
    logger.info("Video saved successfully at: %s", output_path)


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
    margin: int = _MARGIN,
) -> cv2.Mat:
    """Build a per-frame minimap by copying the base canvas and stamping any dots."""
    canvas = base_canvas.copy()
    if triangulated is not None:
        for point in triangulated.points:
            draw_dot(canvas, point, margin=margin)
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


def produce_radar_overlay_video(
    video_path: str,
    triangulation_output: TriangulationOutput,
    output_path: str,
    fps: float | None = None,
    alpha: float = 0.7,
    scale: float = 0.30,
    max_frames: int | None = None,
) -> None:
    """Render a FIFA-style video with a top-down minimap radar overlay at the bottom.

    Streams source frames one at a time so the full video is never held in RAM.

    Parameters:
        - video_path: path to the source camera video
        - triangulation_output: 3D points keyed by frame_index
        - output_path: destination MP4 path
        - fps: frame rate; falls back to triangulation_output.fps if None
        - alpha: minimap opacity (0=invisible, 1=opaque). Default 0.7.
        - scale: minimap width as a fraction of source video width. Default 0.30.
        - max_frames: stop after this many frames; None processes the full video.
    """
    out_fps = fps if fps is not None else triangulation_output.fps

    base = make_base_canvas(margin=0)
    mh, mw = base.shape[:2]
    tri_by_index = {f.frame_index: f for f in triangulation_output.frames}

    frame_iter = iter_video_frames(video_path, max_frames=max_frames)
    try:
        first = next(frame_iter)
    except StopIteration:
        raise ValueError(f"No frames found in {video_path}")

    video_h, video_w = first.shape[:2]
    target_w = int(video_w * scale)
    target_h = int(mh * target_w / mw)
    x0 = (video_w - target_w) // 2
    y0 = video_h - target_h - 10

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, int(out_fps), (video_w, video_h), True)
    if not writer.isOpened():
        raise RuntimeError(f"Could not open VideoWriter for: {output_path}")

    def _annotate(i: int, frame: cv2.Mat) -> cv2.Mat:
        annotated = frame.copy()
        minimap = _render_minimap_frame(base, tri_by_index.get(i), margin=0)
        minimap_resized = cv2.resize(minimap, (target_w, target_h), interpolation=cv2.INTER_AREA)
        overlay_inset(annotated, minimap_resized, x0, y0, alpha)
        return annotated

    try:
        for i, frame in enumerate(itertools.chain([first], frame_iter)):
            writer.write(_annotate(i, frame))
    finally:
        writer.release()
    logger.info("Video saved successfully at: %s", output_path)
