from src.tracking.sort_components import SortTracker
from src.types.tracking import Frame_Detections, TrackingOutput


def apply_sort(
    tracking_output: TrackingOutput,
    *,
    max_iou_distance: float = 0.8,
    max_age: int = 60,
    n_init: int = 2,
) -> TrackingOutput:
    """Run a fresh SORT tracker over a detection-only TrackingOutput.
    Parameters:
    - tracking_output: The input detections to track.
    - max_iou_distance: Maximum IOU distance for matching.
    - max_age: Maximum number of frames to keep "alive" without matches.
    - n_init: Number of consecutive matches needed to confirm a track.
    """
    # Define the SORT tracker with the specified parameters
    tracker = SortTracker(
        max_iou_distance=max_iou_distance,
        max_age=max_age,
        n_init=n_init,
    )
    new_frames: list[Frame_Detections] = []
    for fd in sorted(tracking_output.frames, key=lambda f: f.frame_index):
        tracked = tracker.update(list(fd.detections))
        new_frames.append(Frame_Detections(frame_index=fd.frame_index, detections=tracked))

    return TrackingOutput(
        source=tracking_output.source,
        camera_id=tracking_output.camera_id,
        fps=tracking_output.fps,
        frames=new_frames,
    )
