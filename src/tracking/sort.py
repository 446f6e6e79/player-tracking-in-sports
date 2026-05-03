from src.tracking.sort_components import SortTracker
from src.types.tracking import DetectionOutput, FrameTrackedDetections, TrackingOutput


def apply_sort(
    detection_output: DetectionOutput,
    *,
    max_iou_distance: float = 0.8,
    max_age: int = 60,
    n_init: int = 2,
) -> TrackingOutput:
    """Run a fresh SORT tracker over a DetectionOutput.
    Parameters:
    - detection_output: The input detections to track.
    - max_iou_distance: Maximum IOU distance for matching.
    - max_age: Maximum number of frames to keep "alive" without matches.
    - n_init: Number of consecutive matches needed to confirm a track.
    """
    tracker = SortTracker(
        max_iou_distance=max_iou_distance,
        max_age=max_age,
        n_init=n_init,
    )
    new_frames: list[FrameTrackedDetections] = []
    for fd in sorted(detection_output.frames, key=lambda f: f.frame_index):
        tracked = tracker.update(list(fd.detections))
        new_frames.append(FrameTrackedDetections(frame_index=fd.frame_index, detections=tracked))

    return TrackingOutput(
        source=detection_output.source,
        camera_id=detection_output.camera_id,
        fps=detection_output.fps,
        frames=new_frames,
    )
