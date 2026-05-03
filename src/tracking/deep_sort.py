"""Apply the DeepSORT tracker to a detection-only TrackingOutput.

`apply_deep_sort` runs as post-processing on already-detected frames — the
notebook calls it after merging the two-pass (player + ball) detection results.

This is always the appearance-aware DeepSORT path: per-track gallery of
ResNet18 embeddings + time-since-update matching cascade. The IoU-only SORT
baseline that used to live here has been removed.
"""
import numpy as np

from src.tracking.deep_sort_components import AppearanceEncoder, DeepSortTracker
from src.types.tracking import Frame_Detections, TrackingOutput


def apply_deep_sort(
    tracking_output: TrackingOutput,
    frames: list[np.ndarray],
    *,
    encoder: AppearanceEncoder | None = None,
    max_iou_distance: float = 0.7,
    max_appearance_distance: float = 0.2,
    max_age: int = 30,
    n_init: int = 3,
    feature_budget: int = 100,
) -> TrackingOutput:
    """Run a fresh DeepSortTracker over a detection-only TrackingOutput.

    `frames` is the list of BGR images, indexed by `frame_index`. Frames must
    be in monotonic frame_index order (we sort defensively). Existing
    track_ids on the input are ignored — DeepSORT re-assigns them. Unmatched
    detections are dropped.

    If `encoder` is None, a fresh `AppearanceEncoder` (ResNet18 ImageNet
    weights, auto-picked device) is built internally.
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
