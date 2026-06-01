from __future__ import annotations

from dataclasses import replace

import numpy as np

from src.types.detection import BoundingBox
from src.types.tracking import FrameTrackedDetections, TrackedDetection, TrackingOutput


def smooth_tracking_output(
    tracking: TrackingOutput,
    max_gap: int,
    ema_alpha: float,
) -> TrackingOutput:
    """
    Return a new TrackingOutput with smoothed/interpolated per-track boxes.
    Parameters:
    - tracking: The raw output from the 2D pipeline.
    max_gap: Maximum number of consecutive frames without a *real* detection for
        which the gap is linearly interpolated.  Longer gaps are left as-is.
    ema_alpha: EMA weight in ``[0, 1]``.  1.0 = no smoothing; lower = more smoothing.
    """
    # Index frames by frame_index.
    frame_by_index: dict[int, FrameTrackedDetections] = {
        f.frame_index: f for f in tracking.frames
    }
    frame_indices = sorted(frame_by_index)

    # Group by track_id: list of (frame_index, TrackedDetection), real-only.
    track_real: dict[int, list[tuple[int, TrackedDetection]]] = {}
    for fi in frame_indices:
        for det in frame_by_index[fi].detections:
            if det.confidence > 0.0:
                track_real.setdefault(det.track_id, []).append((fi, det))

    # Build a mutable per-frame detection list (we'll add gap-fills then smooth).
    new_dets: dict[int, list[TrackedDetection]] = {fi: list(frame_by_index[fi].detections) for fi in frame_indices}

    # Step 1: gap interpolation.
    for track_id, real_obs in track_real.items():
        real_obs.sort(key=lambda o: o[0])
        for (fi_a, det_a), (fi_b, det_b) in zip(real_obs, real_obs[1:]):
            gap = fi_b - fi_a - 1
            if gap <= 0 or gap > max_gap:
                continue
            # Linearly interpolate corner coords for the missing frames.
            a = np.array([det_a.bbox.x1, det_a.bbox.y1, det_a.bbox.x2, det_a.bbox.y2])
            b = np.array([det_b.bbox.x1, det_b.bbox.y1, det_b.bbox.x2, det_b.bbox.y2])
            for step in range(1, gap + 1):
                fi_fill = fi_a + step
                if fi_fill not in new_dets:
                    continue
                t = step / (gap + 1)
                coords = (1 - t) * a + t * b
                # Only add the interpolated det if this track isn't already in the frame.
                existing_ids = {d.track_id for d in new_dets[fi_fill]}
                if track_id not in existing_ids:
                    new_dets[fi_fill].append(TrackedDetection(
                        bbox=BoundingBox(x1=coords[0], y1=coords[1], x2=coords[2], y2=coords[3]),
                        confidence=0.0,
                        class_id=det_a.class_id,
                        class_name=det_a.class_name,
                        track_id=track_id,
                    ))

    # Step 2: EMA smoothing per track (forward pass over all frames in order).
    ema_state: dict[int, np.ndarray] = {}  # track_id -> last smoothed corners
    for fi in frame_indices:
        smoothed: list[TrackedDetection] = []
        for det in new_dets[fi]:
            corners = np.array([det.bbox.x1, det.bbox.y1, det.bbox.x2, det.bbox.y2])
            if det.track_id in ema_state:
                corners = ema_alpha * corners + (1.0 - ema_alpha) * ema_state[det.track_id]
            ema_state[det.track_id] = corners
            smoothed.append(TrackedDetection(
                bbox=BoundingBox(x1=corners[0], y1=corners[1], x2=corners[2], y2=corners[3]),
                confidence=det.confidence,
                class_id=det.class_id,
                class_name=det.class_name,
                track_id=det.track_id,
            ))
        new_dets[fi] = smoothed

    new_frames = [
        FrameTrackedDetections(frame_index=fi, detections=new_dets[fi])
        for fi in frame_indices
    ]
    return TrackingOutput(
        source=tracking.source,
        camera_id=tracking.camera_id,
        fps=tracking.fps,
        frames=new_frames,
    )
