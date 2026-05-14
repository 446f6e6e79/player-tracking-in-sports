import numpy as np

from src.tracking.sort_components import AppearanceEncoder, DeepSortTracker
from src.types.detection import DetectionOutput, FrameDetections
from src.types.tracking import FrameTrackedDetections, TrackingOutput


def build_deep_sort_tracker(
    *,
    encoder: AppearanceEncoder | None = None,
    max_iou_distance: float = 0.8,
    max_appearance_distance: float = 0.2,
    max_age: int = 60,
    n_init: int = 2,
    feature_budget: int = 100,
) -> DeepSortTracker:
    """Construct a fresh `DeepSortTracker` with the canonical defaults shared
    by the streaming and batch entry points.
    """
    return DeepSortTracker(
        encoder=encoder if encoder is not None else AppearanceEncoder(),
        max_iou_distance=max_iou_distance,
        max_appearance_distance=max_appearance_distance,
        max_age=max_age,
        n_init=n_init,
        feature_budget=feature_budget,
    )


def step_deep_sort(
    tracker: DeepSortTracker,
    frame_detections: FrameDetections,
    frame: np.ndarray,
) -> FrameTrackedDetections:
    """Run a single DeepSORT update on one frame. Tracker state advances in place."""
    tracked = tracker.update(list(frame_detections.detections), frame)
    return FrameTrackedDetections(frame_index=frame_detections.frame_index, detections=tracked)


def apply_deep_sort(
    detection_output: DetectionOutput,
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
    Run a fresh DeepSortTracker over a DetectionOutput.
    Parameters:
    - detection_output: The input detections to track.
    - frames: The list of BGR images, indexed by `frame_index`.
    - encoder: Optional AppearanceEncoder. If None, a default one is created.
    - max_iou_distance: Maximum IOU distance for matching.
    - max_appearance_distance: Maximum appearance distance for matching.
    - max_age: Maximum number of frames to keep "alive" without matches.
    - n_init: Number of consecutive matches needed to confirm a track.
    """
    tracker = build_deep_sort_tracker(
        encoder=encoder,
        max_iou_distance=max_iou_distance,
        max_appearance_distance=max_appearance_distance,
        max_age=max_age,
        n_init=n_init,
        feature_budget=feature_budget,
    )
    new_frames: list[FrameTrackedDetections] = [
        step_deep_sort(tracker, fd, frames[fd.frame_index])
        for fd in sorted(detection_output.frames, key=lambda f: f.frame_index)
    ]

    return TrackingOutput(
        source=detection_output.source,
        camera_id=detection_output.camera_id,
        fps=detection_output.fps,
        frames=new_frames,
    )
