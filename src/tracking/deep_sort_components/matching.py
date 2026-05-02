"""Track-to-detection association.

Two assignment strategies live here:

1. `min_cost_matching` — pure IoU cost + Hungarian, gated by `max_iou_distance`.
   Used for the tentative-track second pass and as the appearance-free fallback.

2. `matching_cascade` — appearance cosine cost over each track's embedding
   gallery, gated by both IoU and appearance distance, run as a per-`time_since_update`
   cascade so freshly-seen tracks get first pick of detections. This is the part
   that helps identities survive short occlusions.
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


def appearance_cost(
    tracks: list[Track],
    track_indices: list[int],
    detection_features: np.ndarray,
    detection_indices: list[int],
) -> np.ndarray:
    """(T, D) cosine-distance matrix between each track's gallery and the
    selected detection features. Embeddings are assumed L2-normalized, so
    cosine distance is `1 - dot`. The track-side cost is `min` over the
    gallery (best-match-against-history, per the original DeepSORT paper).

    Tracks with empty galleries get cost 1.0 (max distance), forcing the
    gate to reject them — they'll fall through to the IoU pass.
    """
    if not track_indices or not detection_indices:
        return np.empty((len(track_indices), len(detection_indices)))

    det_feats = detection_features[detection_indices]  # (D, dim)
    cost = np.ones((len(track_indices), len(detection_indices)), dtype=float)
    if det_feats.size == 0:
        return cost

    for row, ti in enumerate(track_indices):
        if not tracks[ti].features:
            continue
        track_feats = np.stack(list(tracks[ti].features), axis=0)  # (G, dim)
        sim = track_feats @ det_feats.T                             # (G, D)
        cost[row] = 1.0 - sim.max(axis=0)
    return cost


def matching_cascade(
    tracks: list[Track],
    detections: list[Detection],
    detection_features: np.ndarray,
    track_indices: list[int],
    detection_indices: list[int],
    max_iou_distance: float,
    max_appearance_distance: float,
    cascade_depth: int,
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Run the DeepSORT appearance-driven matching cascade.

    For level = 0..cascade_depth-1, only tracks with `time_since_update == 1+level`
    compete for the remaining detections. Cost is appearance cosine; pairs
    above either gate (IoU or appearance) are excluded before Hungarian.
    """
    matches: list[tuple[int, int]] = []
    remaining_dets = list(detection_indices)

    for level in range(cascade_depth):
        if not remaining_dets:
            break
        level_track_indices = [
            ti for ti in track_indices
            if tracks[ti].time_since_update == 1 + level
        ]
        if not level_track_indices:
            continue

        app_cost = appearance_cost(
            tracks, level_track_indices, detection_features, remaining_dets
        )
        iou_c = iou_cost(tracks, level_track_indices, detections, remaining_dets)

        gated = np.where(
            (iou_c > max_iou_distance) | (app_cost > max_appearance_distance),
            max_appearance_distance + 1e5,
            app_cost,
        )

        row_ind, col_ind = linear_sum_assignment(gated)
        matched_cols: set[int] = set()
        for r, c in zip(row_ind, col_ind):
            if app_cost[r, c] > max_appearance_distance:
                continue
            if iou_c[r, c] > max_iou_distance:
                continue
            matches.append((level_track_indices[r], remaining_dets[c]))
            matched_cols.add(c)

        remaining_dets = [
            remaining_dets[c] for c in range(len(remaining_dets)) if c not in matched_cols
        ]

    matched_track_set = {ti for ti, _ in matches}
    unmatched_tracks = [ti for ti in track_indices if ti not in matched_track_set]
    return matches, unmatched_tracks, remaining_dets
