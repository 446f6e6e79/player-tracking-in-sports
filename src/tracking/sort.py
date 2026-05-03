"""Classical SORT tracker (Bewley et al., 2016).

Pure motion-based tracking: Kalman filter prediction + IoU + Hungarian
assignment. No appearance features, no matching cascade. Useful as the
appearance-free baseline to A/B against `apply_deep_sort` through
`evaluate_tracking()`.

This module reuses the primitives already living in `deep_sort_components/`
(KalmanFilter, Track lifecycle, IoU matching) — the only new code here is
the orchestration loop that runs the IoU two-pass without an encoder.
"""
from src.tracking.deep_sort_components import (
    KalmanFilter,
    Track,
    min_cost_matching,
    xyxy_to_xyah,
)
from src.types.tracking import Detection, Frame_Detections, TrackingOutput


class SortTracker:
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
        """Run one frame of SORT. Returns matched detections with `track_id`
        populated; unmatched detections are dropped (either freshly created
        tentative tracks or noise)."""

        # 1. Predict every existing track one frame forward.
        for track in self.tracks:
            track.predict(self.kf)

        confirmed_indices = [i for i, t in enumerate(self.tracks) if t.is_confirmed()]
        tentative_indices = [i for i, t in enumerate(self.tracks) if t.is_tentative()]
        all_det_indices = list(range(len(detections)))

        # 2. Match confirmed tracks against all detections by IoU.
        matches_a, um_tracks_a, um_dets = min_cost_matching(
            self.tracks, detections,
            confirmed_indices, all_det_indices,
            self.max_iou_distance,
        )
        # 3. Match tentative tracks against the residual.
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

        # Snapshot {track_index -> track_id} BEFORE the prune below shuffles
        # self.tracks indices.
        matched_id_by_index = {ti: self.tracks[ti].track_id for ti, _ in matches}

        # 5. Initiate new tentative tracks.
        for di in um_dets:
            self._initiate_track(detections[di])

        # 6. Drop deleted tracks.
        self.tracks = [t for t in self.tracks if not t.is_deleted()]

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


def apply_sort(
    tracking_output: TrackingOutput,
    *,
    max_iou_distance: float = 0.7,
    max_age: int = 30,
    n_init: int = 3,
) -> TrackingOutput:
    """Run a fresh SortTracker over a detection-only TrackingOutput.

    Frames must be in monotonic frame_index order (we sort defensively).
    Existing track_ids on the input are ignored — SORT re-assigns them.
    Unmatched detections are dropped.
    """
    tracker = SortTracker(
        max_iou_distance=max_iou_distance,
        max_age=max_age,
        n_init=n_init,
    )
    new_frames: list[Frame_Detections] = []
    for fd in sorted(tracking_output.frames, key=lambda f: f.frame_index):
        tracked = tracker.update(list(fd.detections))
        new_frames.append(Frame_Detections(frame_index=fd.frame_index, detections=tracked))

    return TrackingOutput(
        source=tracking_output.source,
        camera_id=tracking_output.camera_id,
        fps=tracking_output.fps,
        frames=new_frames,
    )
