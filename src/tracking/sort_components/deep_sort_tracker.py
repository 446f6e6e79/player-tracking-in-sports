import numpy as np

from src.config import get_config
from src.types.detection import BoundingBox
from src.types.tracking import Detection, TrackedDetection
from src.tracking.sort_components.appearance import AppearanceEncoder
from src.tracking.sort_components.kalman_filter import KalmanFilter
from src.tracking.sort_components.matching import matching_cascade, min_cost_matching
from src.tracking.sort_components.track import Track, xyxy_to_xyah


class DeepSortTracker:
    """
    DeepSORT online tracker for multi-object tracking with appearance re-identification.
    
    DeepSORT uses appearance embeddings (from a pre-trained person re-ID model) to link
    detections across frames, allowing it to recover track identity after occlusions and
    handle crowded scenes better than IoU-only trackers.
    
    Parameters:
        - encoder: AppearanceEncoder instance for extracting visual features from detections.
        - max_iou_distance: Maximum IOU distance for matching (gating threshold).
        - max_appearance_distance: Maximum cosine distance for appearance-based matching.
        - max_age: Maximum number of frames to keep a track alive without matches.
        - n_init: Number of consecutive matches required to promote a tentative track to CONFIRMED.
        - feature_budget: Maximum number of past appearance embeddings to store per track.
    """
    def __init__(
        self,
        encoder: AppearanceEncoder,
        max_iou_distance: float,
        max_appearance_distance: float,
        max_age: int,
        n_init: int,
        feature_budget: int,
    ) -> None:
        self.encoder = encoder
        self.max_iou_distance = max_iou_distance
        self.max_appearance_distance = max_appearance_distance
        self.max_age = max_age
        self.n_init = n_init
        self.feature_budget = feature_budget

        self.kf = KalmanFilter()
        # List of active tracks. Each track has its own state and lifecycle.
        self.tracks: list[Track] = []
        self._next_id = 1

    def update(
        self,
        detections: list[Detection],
        frame: np.ndarray,
    ) -> list[TrackedDetection]:
        """
        Run one step of the DeepSORT tracker for the current frame.
        
        Steps:
            1. Predict every existing track one frame forward (Kalman Filter).
            2. Extract appearance features from all detection crops.
            3. Match confirmed tracks against detections via appearance cascade.
            4. Match tentative tracks against remaining detections by IoU only.
            5. Apply matched updates; mark unmatched tracks as missed.
            6. Initiate new tentative tracks for any remaining detections.
            7. Prune deleted tracks.
            
        Returns a list of TrackedDetections for the current frame with assigned track IDs.
        Unmatched detections (noise or newly tentative) are not returned.
        """

        # 1. Predict every existing track one frame forward.
        for track in self.tracks:
            track.predict(self.kf)

        # 2. Extract appearance features from all detections.
        ball_class_name = get_config().classes.ball_class_name
        if detections:
            boxes = np.asarray([d.get_bbox_tuple() for d in detections], dtype=float)
            features = self.encoder(frame, boxes)
            # Zero out features for ball detections to prevent interference with player matching.
            for i, d in enumerate(detections):
                if d.class_name == ball_class_name:
                    features[i] = 1.0 / features.shape[1] ** 0.5
        else:
            features = np.empty((0, self.encoder.EMBED_DIM), dtype=np.float32)

        # Separate track indices by their state (confirmed vs tentative).
        confirmed_indices = [i for i, t in enumerate(self.tracks) if t.is_confirmed()]
        tentative_indices = [i for i, t in enumerate(self.tracks) if t.is_tentative()]
        all_det_indices = list(range(len(detections)))

        # 3. Match confirmed tracks via appearance cascade.
        # Confirmed tracks have rich appearance galleries, so we use cosine distance gated by IoU.
        matches_a, um_tracks_a, um_dets = matching_cascade(
            self.tracks, detections, features,
            confirmed_indices, all_det_indices,
            self.max_iou_distance, self.max_appearance_distance,
            self.max_age,
        )

        # 4. Match tentative tracks against remaining detections by IoU only.
        # Tentative tracks have only 1-2 samples, so appearance matching is unreliable (canonical DeepSORT step).
        matches_b, um_tracks_b, um_dets = min_cost_matching(
            self.tracks, detections,
            tentative_indices, um_dets,
            self.max_iou_distance,
        )
        
        # Combine matches and unmatched tracks from both stages.
        matches = matches_a + matches_b
        unmatched_tracks = um_tracks_a + um_tracks_b

        # 5. Apply matched updates and mark unmatched tracks as missed.
        for ti, di in matches:
            self.tracks[ti].update(self.kf, detections[di], feature=features[di])
        for ti in unmatched_tracks:
            self.tracks[ti].mark_missed()

        # Snapshot stable data BEFORE the prune below reorders self.tracks indices.
        matched_id_by_index = {ti: self.tracks[ti].track_id for ti, _ in matches}
        # Confirmed-unmatched tracks: capture track objects now so the post-prune loop
        # doesn't index self.tracks with stale (pre-prune) indices → fixes IndexError.
        confirmed_unmatched_snapshot = [
            self.tracks[ti] for ti in um_tracks_a
            if self.tracks[ti].is_confirmed()
            and self.tracks[ti].last_detection.class_name != ball_class_name
        ]

        # 6. Initiate a new tentative track for each unmatched detection.
        for di in um_dets:
            self._initiate_track(detections[di], feature=features[di])

        # 7. Drop all tracks marked as deleted (too old, never confirmed, or tentative one-frame-old).
        self.tracks = [t for t in self.tracks if not t.is_deleted()]

        # Build the per-frame output.
        # (a) Matched detections — re-issued with their stable track_id.
        out: list[TrackedDetection] = []
        for ti, di in matches:
            d = detections[di]
            out.append(TrackedDetection(
                bbox=d.bbox,
                confidence=d.confidence,
                class_id=d.class_id,
                class_name=d.class_name,
                track_id=matched_id_by_index[ti],
            ))

        # (b) Unmatched *confirmed* tracks within max_age — emit the Kalman-predicted
        # box so the player remains visible during short occlusions. The confidence
        # is set to 0 so downstream label-resolution still prefers real detections.
        # Uses the pre-prune snapshot to avoid stale index access into self.tracks.
        for track in confirmed_unmatched_snapshot:
            x1, y1, x2, y2 = track.predicted_xyxy()
            d = track.last_detection
            out.append(TrackedDetection(
                bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
                confidence=0.0,
                class_id=d.class_id,
                class_name=d.class_name,
                track_id=track.track_id,
            ))

        return out

    def _initiate_track(self, detection: Detection, feature: np.ndarray) -> None:
        """Create a new tentative track from a detection and its appearance feature."""
        mean, covariance = self.kf.initiate(xyxy_to_xyah(detection.get_bbox_tuple()))
        self.tracks.append(Track(
            mean=mean,
            covariance=covariance,
            track_id=self._next_id,
            n_init=self.n_init,
            max_age=self.max_age,
            detection=detection,
            feature=feature,
            feature_budget=self.feature_budget,
        ))
        self._next_id += 1
