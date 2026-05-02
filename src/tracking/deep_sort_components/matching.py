"""Track-to-detection association using IoU + Hungarian algorithm.

The full DeepSORT paper combines a Mahalanobis gate (from the Kalman filter)
with an appearance-feature cosine distance, then does an age-cascaded
assignment. We are intentionally simpler here: a single IoU cost + Hungarian,
gated by `max_iou_distance`. That is enough to win on stability vs ByteTrack
in our short clips, and it leaves the appearance head as a clean follow-up.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment

from src.types.tracking import Detection
from src.tracking.deep_sort_components.track import Track
from src.utils.iou import iou


def iou_cost(
    tracks: list[Track],
    track_indices: list[int],
    detections: list[Detection],
    detection_indices: list[int],
) -> np.ndarray:
    """Build a (T, D) cost matrix of `1 - IoU` between predicted track boxes
    and detection boxes. Indices are *positions* in the local arrays, not the
    original list indices."""
    if not track_indices or not detection_indices:
        return np.empty((len(track_indices), len(detection_indices)))

    det_boxes = np.asarray(
        [detections[j].get_bbox_tuple() for j in detection_indices], dtype=float
    )
    cost = np.zeros((len(track_indices), len(detection_indices)))
    for row, ti in enumerate(track_indices):
        track_box = np.asarray(tracks[ti].predicted_xyxy(), dtype=float)
        cost[row] = 1.0 - iou(track_box, det_boxes)
    return cost


def min_cost_matching(
    tracks: list[Track],
    detections: list[Detection],
    track_indices: list[int],
    detection_indices: list[int],
    max_iou_distance: float = 0.7,
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Hungarian assignment over the IoU cost matrix, gated by `max_iou_distance`.

    Returns:
        matches:           list of (track_index, detection_index) pairs (original indices)
        unmatched_tracks:  track indices left without a detection
        unmatched_dets:    detection indices left without a track
    """
    if not track_indices or not detection_indices:
        return [], list(track_indices), list(detection_indices)

    cost = iou_cost(tracks, track_indices, detections, detection_indices)

    # Bias gated entries far above the threshold so linear_sum_assignment
    # never picks them — but still solvable when T != D.
    GATED = max_iou_distance + 1e5
    gated_cost = np.where(cost > max_iou_distance, GATED, cost)

    row_ind, col_ind = linear_sum_assignment(gated_cost)

    matches: list[tuple[int, int]] = []
    matched_rows: set[int] = set()
    matched_cols: set[int] = set()
    for r, c in zip(row_ind, col_ind):
        if cost[r, c] > max_iou_distance:
            continue  # was gated — not a real match
        matches.append((track_indices[r], detection_indices[c]))
        matched_rows.add(r)
        matched_cols.add(c)

    unmatched_tracks = [
        track_indices[r] for r in range(len(track_indices)) if r not in matched_rows
    ]
    unmatched_dets = [
        detection_indices[c] for c in range(len(detection_indices)) if c not in matched_cols
    ]
    return matches, unmatched_tracks, unmatched_dets
