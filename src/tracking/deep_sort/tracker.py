"""DeepSortTracker: per-frame online tracker.

Algorithm (one call to `update(detections)` per video frame):

    1. Predict every existing track one step forward with the Kalman filter.
    2. Match CONFIRMED tracks to detections by IoU + Hungarian (gated).
    3. Match TENTATIVE tracks against the leftover detections (second pass).
    4. Apply matched updates; mark unmatched tracks as missed.
    5. Initiate fresh TENTATIVE tracks for still-unmatched detections.
    6. Drop tracks that ended up in the DELETED state.

This is a deliberate simplification of the full DeepSORT paper:
no appearance ReID network, no Mahalanobis-gated cascade by track age.
"""
import numpy as np

from src.tracking.schema import Detection
from src.tracking.deep_sort.kalman_filter import KalmanFilter
from src.tracking.deep_sort.matching import min_cost_matching
from src.tracking.deep_sort.track import Track, xyxy_to_xyah


class DeepSortTracker:
    def __init__(
        self,
        max_iou_distance: float = 0.7,
        max_age: int = 30,
        n_init: int = 3,
    ) -> None:
        self.max_iou_distance = max_iou_distance
        self.max_age = max_age
        self.n_init = n_init

        self.kf = KalmanFilter()
        self.tracks: list[Track] = []
        self._next_id = 1

    def update(self, detections: list[Detection]) -> list[Detection]:
        """Run one step of the tracker. Returns this frame's matched
        detections with `track_id` populated. Unmatched detections are
        dropped — they're either freshly born tentative tracks (still on
        probation) or noise we don't want to draw."""

        # 1. Predict every existing track one frame forward.
        for track in self.tracks:
            track.predict(self.kf)

        # 2-3. Two-pass association.
        confirmed_indices = [i for i, t in enumerate(self.tracks) if t.is_confirmed()]
        tentative_indices = [i for i, t in enumerate(self.tracks) if t.is_tentative()]
        all_det_indices = list(range(len(detections)))

        matches_a, um_tracks_a, um_dets = min_cost_matching(
            self.tracks, detections,
            confirmed_indices, all_det_indices,
            self.max_iou_distance,
        )
        matches_b, um_tracks_b, um_dets = min_cost_matching(
            self.tracks, detections,
            tentative_indices, um_dets,
            self.max_iou_distance,
        )
        matches = matches_a + matches_b
        unmatched_tracks = um_tracks_a + um_tracks_b

        # 4. Apply matched updates and mark misses.
        for ti, di in matches:
            self.tracks[ti].update(self.kf, detections[di])
        for ti in unmatched_tracks:
            self.tracks[ti].mark_missed()

        # Snapshot {track_index -> track_id} BEFORE pruning so we can stamp
        # output detections without worrying about list reshuffling below.
        matched_id_by_index = {ti: self.tracks[ti].track_id for ti, _ in matches}

        # 5. Initiate new tentative tracks for unmatched detections.
        for di in um_dets:
            self._initiate_track(detections[di])

        # 6. Drop deleted tracks.
        self.tracks = [t for t in self.tracks if not t.is_deleted()]

        # Build the per-frame output: matched detections re-issued with
        # their stable track_id. Order mirrors the matching order.
        out: list[Detection] = []
        for ti, di in matches:
            d = detections[di]
            out.append(Detection(
                bbox=d.bbox,
                confidence=d.confidence,
                class_id=d.class_id,
                class_name=d.class_name,
                track_id=matched_id_by_index[ti],
            ))
        return out

    def _initiate_track(self, detection: Detection) -> None:
        mean, covariance = self.kf.initiate(xyxy_to_xyah(detection.get_bbox_tuple()))
        self.tracks.append(Track(
            mean=mean,
            covariance=covariance,
            track_id=self._next_id,
            n_init=self.n_init,
            max_age=self.max_age,
            detection=detection,
        ))
        self._next_id += 1
