"""DeepSortTracker: per-frame online tracker.

Algorithm (one call to `update(detections, frame)` per video frame):

    1. Predict every existing track one step forward with the Kalman filter.
    2. If an appearance encoder is configured, embed every detection crop.
    3. Match CONFIRMED tracks via the appearance cascade (or fall back to IoU
       when no encoder is set).
    4. Match TENTATIVE tracks against leftover detections by IoU.
    5. Apply matched updates; mark unmatched tracks as missed.
    6. Initiate fresh TENTATIVE tracks for still-unmatched detections.
    7. Drop tracks that ended up in the DELETED state.

Without an encoder this is equivalent to SORT (Kalman + IoU + Hungarian).
With an encoder it adds the per-track appearance gallery and time-since-update
cascade described in the original DeepSORT paper.
"""
import numpy as np

from src.types.tracking import Detection
from src.tracking.deep_sort_components.appearance import AppearanceEncoder
from src.tracking.deep_sort_components.kalman_filter import KalmanFilter
from src.tracking.deep_sort_components.matching import matching_cascade, min_cost_matching
from src.tracking.deep_sort_components.track import Track, xyxy_to_xyah


class DeepSortTracker:
    def __init__(
        self,
        encoder: AppearanceEncoder | None = None,
        max_iou_distance: float = 0.7,
        max_appearance_distance: float = 0.2,
        max_age: int = 30,
        n_init: int = 3,
        feature_budget: int = 100,
    ) -> None:
        self.encoder = encoder
        self.max_iou_distance = max_iou_distance
        self.max_appearance_distance = max_appearance_distance
        self.max_age = max_age
        self.n_init = n_init
        self.feature_budget = feature_budget

        self.kf = KalmanFilter()
        self.tracks: list[Track] = []
        self._next_id = 1

    def update(
        self,
        detections: list[Detection],
        frame: np.ndarray | None = None,
    ) -> list[Detection]:
        """Run one step of the tracker. Returns this frame's matched
        detections with `track_id` populated. Unmatched detections are
        dropped — they're either freshly born tentative tracks (still on
        probation) or noise we don't want to draw."""

        # 1. Predict every existing track one frame forward.
        for track in self.tracks:
            track.predict(self.kf)

        # 2. Embed detection crops if we have both an encoder and a frame.
        features: np.ndarray | None = None
        if self.encoder is not None and frame is not None and detections:
            boxes = np.asarray([d.get_bbox_tuple() for d in detections], dtype=float)
            features = self.encoder(frame, boxes)

        confirmed_indices = [i for i, t in enumerate(self.tracks) if t.is_confirmed()]
        tentative_indices = [i for i, t in enumerate(self.tracks) if t.is_tentative()]
        all_det_indices = list(range(len(detections)))

        # 3. Confirmed tracks: appearance cascade if we have features, else IoU.
        if features is not None:
            matches_a, um_tracks_a, um_dets = matching_cascade(
                self.tracks, detections, features,
                confirmed_indices, all_det_indices,
                self.max_iou_distance, self.max_appearance_distance,
                self.max_age,
            )
        else:
            matches_a, um_tracks_a, um_dets = min_cost_matching(
                self.tracks, detections,
                confirmed_indices, all_det_indices,
                self.max_iou_distance,
            )

        # 4. Tentative tracks always go through IoU — galleries are too thin
        # to trust appearance for newborn tracks.
        matches_b, um_tracks_b, um_dets = min_cost_matching(
            self.tracks, detections,
            tentative_indices, um_dets,
            self.max_iou_distance,
        )
        matches = matches_a + matches_b
        unmatched_tracks = um_tracks_a + um_tracks_b

        # 5. Apply matched updates and mark misses.
        for ti, di in matches:
            feat = features[di] if features is not None else None
            self.tracks[ti].update(self.kf, detections[di], feature=feat)
        for ti in unmatched_tracks:
            self.tracks[ti].mark_missed()

        # Snapshot {track_index -> track_id} BEFORE pruning so we can stamp
        # output detections without worrying about list reshuffling below.
        matched_id_by_index = {ti: self.tracks[ti].track_id for ti, _ in matches}

        # 6. Initiate new tentative tracks for unmatched detections.
        for di in um_dets:
            feat = features[di] if features is not None else None
            self._initiate_track(detections[di], feature=feat)

        # 7. Drop deleted tracks.
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

    def _initiate_track(
        self,
        detection: Detection,
        feature: np.ndarray | None = None,
    ) -> None:
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
