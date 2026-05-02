"""Apply the DeepSORT-style tracker to a detection-only TrackingOutput.

`apply_deep_sort` runs as post-processing on already-detected frames — the
notebook calls it after merging the two-pass (player + ball) detection
results.

Two modes:

  - IoU baseline (default): pass only the TrackingOutput. Behaves like SORT —
    Kalman + IoU + Hungarian. Identical to the previous implementation, kept so
    we have a stable baseline to compare against in `evaluate_tracking`.

  - Appearance-aware: pass `frames` (the BGR images) and an `encoder`. The
    tracker then runs the per-track gallery + cascade described in the
    DeepSORT paper, which keeps identities stable through short occlusions.
"""
import numpy as np

from src.tracking.deep_sort_components import AppearanceEncoder, DeepSortTracker
from src.types.tracking import Frame_Detections, TrackingOutput


def apply_deep_sort(
    tracking_output: TrackingOutput,
    *,
    frames: list[np.ndarray] | None = None,
    encoder: AppearanceEncoder | None = None,
    max_iou_distance: float = 0.7,
    max_appearance_distance: float = 0.2,
    max_age: int = 30,
    n_init: int = 3,
    feature_budget: int = 100,
) -> TrackingOutput:
    """Run a fresh DeepSortTracker over a detection-only TrackingOutput.

    Frames must be in monotonic frame_index order (we sort defensively).
    Existing track_ids on the input are ignored — DeepSORT re-assigns them.
    Unmatched detections are dropped.

    When `encoder` is provided, `frames` must also be provided so that the
    tracker can pull image crops for each detection.
    """
    if encoder is not None and frames is None:
        raise ValueError(
            "apply_deep_sort: `frames` must be provided when `encoder` is set"
        )

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
        frame_img = frames[fd.frame_index] if frames is not None else None
        tracked = tracker.update(list(fd.detections), frame=frame_img)
        new_frames.append(Frame_Detections(frame_index=fd.frame_index, detections=tracked))

    return TrackingOutput(
        source=tracking_output.source,
        camera_id=tracking_output.camera_id,
        fps=tracking_output.fps,
        frames=new_frames,
    )
