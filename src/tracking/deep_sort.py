import numpy as np

from src.tracking.sort_components import AppearanceEncoder, DeepSortTracker
from src.types.tracking import Frame_Detections, TrackingOutput


def apply_deep_sort(
    tracking_output: TrackingOutput,
    frames: list[np.ndarray],
    *,
    encoder: AppearanceEncoder | None = None,
    max_iou_distance: float = 0.8,
    max_appearance_distance: float = 0.2,
    max_age: int = 60,
    n_init: int = 2,
    feature_budget: int = 100,
) -> TrackingOutput:
    """
    Run a fresh DeepSortTracker over a detection-only TrackingOutput.
    Parameters:
    - tracking_output: The input detections to track.
    - frames: The list of BGR images, indexed by `frame_index`.
    - encoder: Optional AppearanceEncoder. If None, a default one is created.
    - max_iou_distance: Maximum IOU distance for matching.
    - max_appearance_distance: Maximum appearance distance for matching.
    - max_age: Maximum number of frames to keep "alive" without matches.
    - n_init: Number of consecutive matches needed to confirm a track.
    """
    if encoder is None:
        encoder = AppearanceEncoder()

    tracker = DeepSortTracker(
        encoder=encoder,
        max_iou_distance=max_iou_distance,
        max_appearance_distance=max_appearance_distance,
        max_age=max_age,
        n_init=n_init,
        feature_budget=feature_budget,
    )
    new_frames: list[Frame_Detections] = []
    for fd in sorted(tracking_output.frames, key=lambda f: f.frame_index):
        tracked = tracker.update(list(fd.detections), frames[fd.frame_index])
        new_frames.append(Frame_Detections(frame_index=fd.frame_index, detections=tracked))

    return TrackingOutput(
        source=tracking_output.source,
        camera_id=tracking_output.camera_id,
        fps=tracking_output.fps,
        frames=new_frames,
    )
