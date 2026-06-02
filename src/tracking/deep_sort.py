import numpy as np

from src.config import get_config
from src.tracking.sort_components import AppearanceEncoder, DeepSortTracker
from src.types.detection import DetectionOutput, FrameDetections
from src.types.tracking import FrameTrackedDetections, TrackingOutput


def _make_encoder() -> AppearanceEncoder:
    """Build a default AppearanceEncoder from config."""
    from src.paths.model_paths import OSNET_WEIGHTS_PATH
    cfg = get_config()
    return AppearanceEncoder(
        weights_path=OSNET_WEIGHTS_PATH,
        model_name=cfg.models.osnet_model_name,
        input_size=cfg.tracking.appearance.input_size,
        embed_dim=cfg.tracking.appearance.embed_dim,
    )


def build_deep_sort_tracker(
    *,
    encoder: AppearanceEncoder | None = None,
    max_iou_distance: float,
    max_appearance_distance: float,
    max_age: int,
    n_init: int,
    feature_budget: int,
) -> DeepSortTracker:
    """Construct a fresh ``DeepSortTracker``."""
    return DeepSortTracker(
        encoder=encoder if encoder is not None else _make_encoder(),
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
    low_conf_detections: list | None = None,
) -> FrameTrackedDetections:
    """Run a single DeepSORT update on one frame. Tracker state advances in place."""
    tracked = tracker.update(
        list(frame_detections.detections), frame,
        low_conf_detections=low_conf_detections,
    )
    return FrameTrackedDetections(frame_index=frame_detections.frame_index, detections=tracked)


def apply_deep_sort(
    detection_output: DetectionOutput,
    frames: list[np.ndarray],
    *,
    encoder: AppearanceEncoder | None = None,
    max_iou_distance: float,
    max_appearance_distance: float,
    max_age: int,
    n_init: int,
    feature_budget: int,
) -> TrackingOutput:
    """
    Run a fresh DeepSortTracker over a DetectionOutput.
    Parameters:
    - detection_output: The input detections to track.
    - frames: The list of BGR images, indexed by ``frame_index``.
    - encoder: Optional AppearanceEncoder. If None, one is built from config.
    - max_iou_distance: Maximum IOU distance for matching.
    - max_appearance_distance: Maximum appearance distance for matching.
    - max_age: Maximum number of frames to keep "alive" without matches.
    - n_init: Number of consecutive matches needed to confirm a track.
    - feature_budget: Maximum appearance features stored per track.
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
