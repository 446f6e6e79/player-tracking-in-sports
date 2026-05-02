"""A single tracked object: Kalman state + lifecycle counters.

Lifecycle:
    TENTATIVE  : freshly created, not yet trusted (might be a false positive)
                 -> CONFIRMED after `n_init` consecutive matched frames
                 -> DELETED on the very first miss
    CONFIRMED  : trusted track. Coasts (Kalman-predicted only) when not matched.
                 -> DELETED once `time_since_update` exceeds `max_age`
    DELETED    : removed from the tracker at the end of the step.
"""
from collections import deque
from enum import Enum

import numpy as np

from src.types.tracking import Detection
from src.tracking.deep_sort_components.kalman_filter import KalmanFilter


class TrackState(Enum):
    TENTATIVE = 1
    CONFIRMED = 2
    DELETED = 3


def xyxy_to_xyah(bbox) -> np.ndarray:
    """Convert (x1, y1, x2, y2) -> (cx, cy, a, h) measurement vector."""
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    cx = x1 + w / 2.0
    cy = y1 + h / 2.0
    a = w / max(h, 1e-6)
    return np.array([cx, cy, a, h], dtype=float)


def xyah_to_xyxy(state: np.ndarray) -> tuple[float, float, float, float]:
    """Convert state[:4] = (cx, cy, a, h) -> (x1, y1, x2, y2)."""
    cx, cy, a, h = state[:4]
    w = a * h
    return (cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0)


class Track:
    def __init__(
        self,
        mean: np.ndarray,
        covariance: np.ndarray,
        track_id: int,
        n_init: int,
        max_age: int,
        detection: Detection,
        feature: np.ndarray | None = None,
        feature_budget: int = 100,
    ) -> None:
        self.mean = mean
        self.covariance = covariance
        self.track_id = track_id

        self.hits = 1                  # total matched frames
        self.age = 1                   # total frames since creation
        self.time_since_update = 0     # frames since last match
        self.state = TrackState.TENTATIVE

        self.n_init = n_init
        self.max_age = max_age
        self.last_detection = detection  # carries class_id / class_name into output

        # Rolling gallery of recent appearance embeddings (L2-normalized).
        # Empty when the tracker is run without an appearance encoder.
        self.features: deque[np.ndarray] = deque(maxlen=feature_budget)
        if feature is not None:
            self.features.append(feature)

    def predicted_xyxy(self) -> tuple[float, float, float, float]:
        """Current Kalman-predicted bbox in xyxy."""
        return xyah_to_xyxy(self.mean)

    def predict(self, kf: KalmanFilter) -> None:
        """Advance the filter one step and bump bookkeeping counters."""
        self.mean, self.covariance = kf.predict(self.mean, self.covariance)
        self.age += 1
        self.time_since_update += 1

    def update(
        self,
        kf: KalmanFilter,
        detection: Detection,
        feature: np.ndarray | None = None,
    ) -> None:
        """Apply a matched detection: Kalman update + reset miss counter."""
        measurement = xyxy_to_xyah(detection.get_bbox_tuple())
        self.mean, self.covariance = kf.update(self.mean, self.covariance, measurement)

        self.hits += 1
        self.time_since_update = 0
        self.last_detection = detection
        if feature is not None:
            self.features.append(feature)

        # Promote out of tentative once we've seen enough consecutive hits.
        if self.state == TrackState.TENTATIVE and self.hits >= self.n_init:
            self.state = TrackState.CONFIRMED

    def mark_missed(self) -> None:
        """No detection matched this track this frame."""
        if self.state == TrackState.TENTATIVE:
            # Tentative tracks are likely false positives -> kill immediately.
            self.state = TrackState.DELETED
        elif self.time_since_update > self.max_age:
            self.state = TrackState.DELETED

    def is_tentative(self) -> bool:
        return self.state == TrackState.TENTATIVE

    def is_confirmed(self) -> bool:
        return self.state == TrackState.CONFIRMED

    def is_deleted(self) -> bool:
        return self.state == TrackState.DELETED
