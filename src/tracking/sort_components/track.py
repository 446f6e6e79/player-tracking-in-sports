from collections import deque
from enum import Enum

import numpy as np

from src.types.tracking import Detection
from src.tracking.sort_components.kalman_filter import KalmanFilter


class TrackState(Enum):
    """
    Enumeration type for the single object track state.
    Each track can be in one of three states:
    - Tentative: The track is newly created and not confirmed until it has enough hits.
    - Confirmed: The track has been matched with detections for at least `n_init` 
      consecutive frames and is considered a valid track.
    - Deleted: The track has been marked for deletion due to too 
      many missed matches or being a false positive (if it was never confirmed).
    """
    TENTATIVE = 1
    CONFIRMED = 2
    DELETED = 3


def xyxy_to_xyah(bbox) -> np.ndarray:
    """Convert (x1, y1, x2, y2) -> (cx, cy, a, h) measurement vector."""
    x1, y1, x2, y2 = bbox
    # Get height and width of the bounding box
    w = x2 - x1
    h = y2 - y1
    
    # Compute centers and aspect ratio
    cx = x1 + w / 2.0
    cy = y1 + h / 2.0
    a = w / max(h, 1e-6)
    return np.array([cx, cy, a, h], dtype=float)


def xyah_to_xyxy(state: np.ndarray) -> tuple[float, float, float, float]:
    """Convert state[:4] = (cx, cy, a, h) -> (x1, y1, x2, y2)."""
    # Extract center, aspect ratio, and height from the state vector
    cx, cy, a, h = state[:4]
    # Reverse the aspect ratio to get width, then compute corners
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
        """
        Represents a single object track with its state and lifecycle.
        Parameters:
        - mean: Kalman filter mean state vector (8D: [cx, cy, a, h, vx, vy, va, vh]).
        - covariance: Kalman filter covariance matrix (8x8).
        - track_id: Unique integer ID for this track.
        - n_init: Number of consecutive matches needed to confirm a track.
        - max_age: Maximum number of frames to keep "alive" without matches.
        - detection: The initial Detection that created this track (carries class_id / class_name).
        - feature: Optional initial appearance feature vector (L2-normalized).
        - feature_budget: Maximum number of recent features to store for this track.
        """
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
        # Update the mean and covariance using the Kalman filter's predict step.
        self.mean, self.covariance = kf.predict(self.mean, self.covariance)
        # Increment age and time since update, which are used for track management.
        self.age += 1
        self.time_since_update += 1

    def update(
        self,
        kf: KalmanFilter,
        detection: Detection,
        feature: np.ndarray | None = None,
    ) -> None:
        """
        Perform Kalman filter measurement update and update track state.
        Parameters:
            - kf: The KalmanFilter instance to use for the update step.
            - detection: The Detection that matched this track (carries class_id / class_name).
            - feature: Optional appearance feature vector for this detection (L2-normalized).
        """
        # Convert the detection's bounding box to the measurement space (cx, cy, a, h)
        measurement = xyxy_to_xyah(detection.get_bbox_tuple())
        # Update the mean and covariance using the Kalman filter's update step with the new measurement
        self.mean, self.covariance = kf.update(self.mean, self.covariance, measurement)

        self.hits += 1
        self.time_since_update = 0
        self.last_detection = detection
        
        # Append the new feature to the track's feature gallery if provided
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
