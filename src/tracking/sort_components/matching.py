import numpy as np
from scipy.optimize import linear_sum_assignment

from src.types.tracking import Detection
from src.tracking.sort_components.track import Track
from src.utils.iou import iou


def _iou_cost(
    tracks: list[Track],
    track_indices: list[int],
    detections: list[Detection],
    detection_indices: list[int],
) -> np.ndarray:
    """
    Build a (T, D) matrix of IoU distances between tracks and detections.
    cost[i, j] is the IoU distance between tracks[track_indices[i]] and
    detections[detection_indices[j]]. 
    The iou distance is 1 - iou, so in [0, 1], where 0 is a perfect match and 1 is no overlap.
    
    Parameters:
        - tracks: list of all current tracks
        - track_indices: indices of tracks to consider (rows)
        - detections: list of all current detections
        - detection_indices: indices of detections to consider (columns)
    Returns:
        - cost: (T, D) array of IoU distances (1 - IoU) between tracks and detections
    """
    if not track_indices or not detection_indices:
        return np.empty((len(track_indices), len(detection_indices)))

    # Extract the bounding boxes for the selected detections
    det_boxes = np.asarray(
        [detections[j].get_bbox_tuple() for j in detection_indices], dtype=float
    )

    cost = np.zeros((len(track_indices), len(detection_indices)))
    
    # For each track, compute the IoU distance to each detection and fill the cost matrix
    for row, track_index in enumerate(track_indices):
        # Get the predicted bounding box for the track in (x1, y1, x2, y2) format
        track_box = np.asarray(tracks[track_index].predicted_xyxy(), dtype=float)
        cost[row] = 1.0 - iou(track_box, det_boxes)
    return cost


def min_cost_matching(
    tracks: list[Track],
    detections: list[Detection],
    track_indices: list[int],
    detection_indices: list[int],
    max_iou_distance: float,
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """
    Hungarian matching based on the IoU distance matrix.
    Match tracks to detections based on the minimum IoU distance, with gating by max_iou_distance.
    Parameters:
        - tracks: list of all current tracks
        - detections: list of all current detections
        - track_indices: indices of tracks to consider (rows)
        - detection_indices: indices of detections to consider (columns)
        - max_iou_distance: maximum allowed IoU distance for a valid match
    Returns:
        - matches: list of (track_index, detection_index) pairs for matched tracks and detections
        - unmatched_tracks: list of track indices that were not matched    
        - unmatched_detections: list of detection indices that were not matched
    """
    if not track_indices or not detection_indices:
        return [], list(track_indices), list(detection_indices)

    # Compute the IoU distance matrix between the selected tracks and detections
    cost_matrix = _iou_cost(tracks, track_indices, detections, detection_indices)

    # Apply gating: set cost to a large value for pairs that exceed the max_iou_distance, so they won't be matched
    GATED = max_iou_distance + 1e5
    gated_cost_matrix = np.where(cost_matrix > max_iou_distance, GATED, cost_matrix)

    # Solve the linear assignment problem (Hungarian algorithm) on the gated cost matrix
    # row_ind and col_ind are the indices of the matched pairs in the cost matrix
    row_ind, col_ind = linear_sum_assignment(gated_cost_matrix)

    matches: list[tuple[int, int]] = []
    matched_rows: set[int] = set()
    matched_cols: set[int] = set()

    for row, col in zip(row_ind, col_ind):
        # Skip gated pairs
        if cost_matrix[row, col] > max_iou_distance:
            continue
        # Add the valid match (track_index, detection_index) to the matches list
        matches.append((track_indices[row], detection_indices[col]))
        matched_rows.add(row)
        matched_cols.add(col)

    unmatched_tracks = [
        track_indices[row] for row in range(len(track_indices)) if row not in matched_rows
    ]
    unmatched_dets = [
        detection_indices[col] for col in range(len(detection_indices)) if col not in matched_cols
    ]
    return matches, unmatched_tracks, unmatched_dets


def appearance_cost(
    tracks: list[Track],
    track_indices: list[int],
    detection_features: np.ndarray,
    detection_indices: list[int],
) -> np.ndarray:
    """
    (T, D) cosine-distance matrix between each track's gallery (historical aspect features)
    and the current detection features. Lower cost means more similar.
    Embeddings are assumed L2-normalized, so cosine distance is `1 - dot_product`. 
    
    The track-side cost is `min` over the gallery (best-match-against-history, 
    per the original DeepSORT paper).

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
    """
    Run the DeepSORT appearance-driven matching cascade.

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
        iou_c = _iou_cost(tracks, level_track_indices, detections, remaining_dets)

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
