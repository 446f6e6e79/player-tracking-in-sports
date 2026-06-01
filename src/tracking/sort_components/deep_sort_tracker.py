from collections import deque

import numpy as np

from src.config import get_config
from src.types.detection import BoundingBox
from src.types.tracking import Detection, TrackedDetection
from src.tracking.sort_components.appearance import AppearanceEncoder
from src.tracking.sort_components.kalman_filter import KalmanFilter
from src.tracking.sort_components.matching import matching_cascade, min_cost_matching
from src.tracking.sort_components.track import Track, TrackState, xyxy_to_xyah
from src.utils.iou import iou


class DeepSortTracker:
    """
    DeepSORT online tracker for multi-object tracking with appearance re-identification.

    Extensions over the original DeepSORT:
      - NSA Kalman: measurement noise R is scaled by (1 - confidence)^2 so the
        filter trusts high-confidence detections more (less jitter).
      - ByteTrack second pass: after the main cascade, confirmed unmatched tracks
        are IoU-matched against low-confidence player detections to recover tracks
        during partial occlusions without polluting the appearance gallery.
      - Dead-track re-ID gallery: appearance galleries from recently-dead tracks
        are kept for `dead_gallery_max_age` frames. When a new detection matches a
        dead gallery entry (cosine distance < threshold), the original track ID is
        resurrected instead of a new ID being assigned.
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
        self.tracks: list[Track] = []
        self._next_id = 1
        # dead_gallery: track_id -> {features, class_name, frame}
        self._dead_gallery: dict[int, dict] = {}
        self._frame: int = 0

    def update(
        self,
        detections: list[Detection],
        frame: np.ndarray,
        low_conf_detections: list[Detection] | None = None,
    ) -> list[TrackedDetection]:
        """
        Run one step of the DeepSORT tracker for the current frame.

        Parameters:
            - detections: high-confidence detections for this frame.
            - frame: BGR image array (for appearance encoding).
            - low_conf_detections: optional list of low-confidence player detections
              used in the ByteTrack second-pass IoU matching.
        """
        self._frame += 1
        low_conf_detections = low_conf_detections or []

        # 1. Predict every existing track one frame forward.
        for track in self.tracks:
            track.predict(self.kf)

        # 2. Extract appearance features from all detections.
        cfg = get_config()
        ds_cfg = cfg.tracking.deep_sort
        ball_class_name = cfg.classes.ball_class_name
        if detections:
            boxes = np.asarray([d.get_bbox_tuple() for d in detections], dtype=float)
            features = self.encoder(frame, boxes)
            for i, d in enumerate(detections):
                if d.class_name == ball_class_name:
                    features[i] = 1.0 / features.shape[1] ** 0.5
        else:
            features = np.empty((0, self.encoder.EMBED_DIM), dtype=np.float32)

        confirmed_indices = [i for i, t in enumerate(self.tracks) if t.is_confirmed()]
        tentative_indices = [i for i, t in enumerate(self.tracks) if t.is_tentative()]
        all_det_indices = list(range(len(detections)))

        # 3. Match confirmed tracks via appearance cascade.
        matches_a, um_tracks_a, um_dets = matching_cascade(
            self.tracks, detections, features,
            confirmed_indices, all_det_indices,
            self.max_iou_distance, self.max_appearance_distance,
            self.max_age,
        )

        # 4. Match tentative tracks against remaining detections by IoU only.
        matches_b, um_tracks_b, um_dets = min_cost_matching(
            self.tracks, detections,
            tentative_indices, um_dets,
            self.max_iou_distance,
        )

        matches = matches_a + matches_b
        unmatched_tracks = um_tracks_a + um_tracks_b

        # 5. Apply matched updates (NSA KF: confidence flows through track.update).
        for ti, di in matches:
            self.tracks[ti].update(self.kf, detections[di], feature=features[di])
        for ti in unmatched_tracks:
            self.tracks[ti].mark_missed()

        # 5b. ByteTrack second pass: IoU-only match of confirmed unmatched tracks
        # against low-confidence detections (no gallery update to avoid noisy crops).
        bytetrack_out: list[TrackedDetection] = []
        bytetrack_matched: set[int] = set()
        if low_conf_detections:
            lc_candidates = [ti for ti in um_tracks_a if not self.tracks[ti].is_deleted()]
            if lc_candidates:
                matches_c, _, _ = min_cost_matching(
                    self.tracks, low_conf_detections,
                    lc_candidates, list(range(len(low_conf_detections))),
                    self.max_iou_distance,
                )
                for ti, di in matches_c:
                    self.tracks[ti].update(self.kf, low_conf_detections[di], feature=None)
                    bytetrack_matched.add(ti)
                    d = low_conf_detections[di]
                    bytetrack_out.append(TrackedDetection(
                        bbox=d.bbox, confidence=0.0,
                        class_id=d.class_id, class_name=d.class_name,
                        track_id=self.tracks[ti].track_id,
                    ))

        # Snapshot before prune — stale indices become invalid after filtering.
        matched_id_by_index = {ti: self.tracks[ti].track_id for ti, _ in matches}
        confirmed_unmatched_snapshot = [
            self.tracks[ti] for ti in um_tracks_a
            if self.tracks[ti].is_confirmed()
            and self.tracks[ti].last_detection.class_name != ball_class_name
            and self.tracks[ti].time_since_update <= ds_cfg.coast_age
            and ti not in bytetrack_matched
        ]

        # 5c. Dead gallery: capture dying confirmed tracks, expire stale entries.
        for track in self.tracks:
            if track.is_deleted() and track.hits >= track.n_init and track.features:
                self._dead_gallery[track.track_id] = {
                    'features': list(track.features),
                    'class_name': track.last_detection.class_name,
                    'frame': self._frame,
                }
        stale = [tid for tid, e in self._dead_gallery.items()
                 if self._frame - e['frame'] > ds_cfg.dead_gallery_max_age]
        for tid in stale:
            del self._dead_gallery[tid]

        # 6. For each unmatched detection, try re-ID resurrection before initiating fresh.
        resurrected_out: list[TrackedDetection] = []
        still_unmatched: list[int] = []
        for di in um_dets:
            det = detections[di]
            feat = features[di]
            best_id, best_dist = None, ds_cfg.dead_gallery_max_dist
            for tid, entry in self._dead_gallery.items():
                if entry['class_name'] != det.class_name:
                    continue
                gallery_mean = np.mean(np.stack(entry['features']), axis=0)
                dist = 1.0 - float(gallery_mean @ feat)
                if dist < best_dist:
                    best_dist, best_id = dist, tid
            if best_id is not None:
                self._resurrect_track(best_id, self._dead_gallery[best_id]['features'], det, feat)
                del self._dead_gallery[best_id]
                resurrected_out.append(TrackedDetection(
                    bbox=det.bbox, confidence=det.confidence,
                    class_id=det.class_id, class_name=det.class_name,
                    track_id=best_id,
                ))
            else:
                still_unmatched.append(di)

        for di in still_unmatched:
            self._initiate_track(detections[di], feature=features[di])

        # 7. Prune deleted tracks.
        self.tracks = [t for t in self.tracks if not t.is_deleted()]

        # Build output.
        out: list[TrackedDetection] = []

        # (a) Matched high-confidence detections.
        for ti, di in matches:
            d = detections[di]
            out.append(TrackedDetection(
                bbox=d.bbox, confidence=d.confidence,
                class_id=d.class_id, class_name=d.class_name,
                track_id=matched_id_by_index[ti],
            ))

        # (b) ByteTrack-recovered detections (low-conf, position update only).
        out.extend(bytetrack_out)

        # (c) Resurrected detections (from dead gallery, real confidence).
        out.extend(resurrected_out)

        # (d) Coasted predictions for briefly occluded confirmed tracks.
        matched_boxes = (
            np.array([[d.bbox.x1, d.bbox.y1, d.bbox.x2, d.bbox.y2]
                      for _, di in matches for d in [detections[di]]], dtype=np.float32)
            if matches else np.empty((0, 4), dtype=np.float32)
        )
        frame_h, frame_w = frame.shape[:2]
        for track in confirmed_unmatched_snapshot:
            x1, y1, x2, y2 = track.predicted_xyxy()
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            if not (0 <= cx < frame_w and 0 <= cy < frame_h):
                continue
            if len(matched_boxes) > 0:
                if iou(np.array([x1, y1, x2, y2], dtype=np.float32), matched_boxes).max() > ds_cfg.coast_max_iou:
                    continue
            d = track.last_detection
            out.append(TrackedDetection(
                bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
                confidence=0.0, class_id=d.class_id,
                class_name=d.class_name, track_id=track.track_id,
            ))

        return out

    def _initiate_track(self, detection: Detection, feature: np.ndarray) -> None:
        mean, covariance = self.kf.initiate(xyxy_to_xyah(detection.get_bbox_tuple()))
        self.tracks.append(Track(
            mean=mean, covariance=covariance,
            track_id=self._next_id,
            n_init=self.n_init, max_age=self.max_age,
            detection=detection, feature=feature,
            feature_budget=self.feature_budget,
        ))
        self._next_id += 1

    def _resurrect_track(
        self,
        track_id: int,
        gallery: list[np.ndarray],
        detection: Detection,
        feature: np.ndarray,
    ) -> None:
        """Reinstate a dead track with its old ID, seeding from the saved gallery."""
        mean, covariance = self.kf.initiate(xyxy_to_xyah(detection.get_bbox_tuple()))
        track = Track(
            mean=mean, covariance=covariance,
            track_id=track_id,
            n_init=self.n_init, max_age=self.max_age,
            detection=detection, feature=None,
            feature_budget=self.feature_budget,
        )
        track.features = deque(gallery, maxlen=self.feature_budget)
        track.features.append(feature)
        track.state = TrackState.CONFIRMED
        track.hits = self.n_init
        self.tracks.append(track)
