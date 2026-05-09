import cv2

from src.types.tracking import DetectionOutput, TrackingOutput
from src.utils.drawing import draw_detections, draw_tracked_detections
from src.utils.video_io import save_video


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
