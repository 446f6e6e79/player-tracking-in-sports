"""Apply the DeepSORT-style tracker to a detection-only TrackingOutput.

`apply_deep_sort` runs as post-processing on already-detected frames — the
notebook calls it after merging the two-pass (player + ball) detection
results.
"""
from src.tracking.deep_sort_components import DeepSortTracker
from src.types.tracking import Frame_Detections, TrackingOutput


def apply_deep_sort(
    tracking_output: TrackingOutput,
    *,
    max_iou_distance: float = 0.7,
    max_age: int = 30,
    n_init: int = 3,
) -> TrackingOutput:
    """Run a fresh DeepSortTracker over a detection-only TrackingOutput.

    Frames must be in monotonic frame_index order (we sort defensively).
    Existing track_ids on the input are ignored — DeepSORT re-assigns them.
    Unmatched detections are dropped.
    """
    tracker = DeepSortTracker(
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
