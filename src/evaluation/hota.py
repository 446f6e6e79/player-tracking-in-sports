import numpy as np
import trackeval

from src.types.evaluation import HOTAMetrics
from src.types.tracking import Detection, TrackingOutput


def _iou_matrix(gt_dets: list[Detection], pred_dets: list[Detection]) -> np.ndarray:
    """Compute (N_gt, N_pred) IoU matrix.
    Parameters:
        - gt_dets: ground-truth detections for this frame.
        - pred_dets: predicted detections for this frame.
    Returns:
        - (N_gt, N_pred) float array with IoU values in [0, 1].
    """
    # Convert list of Detection objects to (N, 4) array of [x1, y1, x2, y2] for GT and predicted detections
    gt   = np.array([[d.bbox.x1, d.bbox.y1, d.bbox.x2, d.bbox.y2] for d in gt_dets],   dtype=float)
    pred = np.array([[d.bbox.x1, d.bbox.y1, d.bbox.x2, d.bbox.y2] for d in pred_dets], dtype=float)

    # Vectorised intersection: broadcast (N, 1, 4) and (1, M, 4)
    intersection_x1 = np.maximum(gt[:, None, 0], pred[None, :, 0])
    intersection_y1 = np.maximum(gt[:, None, 1], pred[None, :, 1])
    intersection_x2 = np.minimum(gt[:, None, 2], pred[None, :, 2])
    intersection_y2 = np.minimum(gt[:, None, 3], pred[None, :, 3])

    # Intersection area is width * height of the overlapping region, clipped to zero if no overlap
    intersection    = np.maximum(0.0, intersection_x2 - intersection_x1) * np.maximum(0.0, intersection_y2 - intersection_y1)

    # Union area is sum of individual areas minus intersection
    area_gt   = (gt[:, 2]   - gt[:, 0])   * (gt[:, 3]   - gt[:, 1])
    area_pred = (pred[:, 2] - pred[:, 0]) * (pred[:, 3] - pred[:, 1])
    union     = area_gt[:, None] + area_pred[None, :] - intersection

    # IoU is intersection over union, defined to be 0 when union is zero (should not happen with valid boxes)
    return np.where(union > 0, intersection / union, 0.0)


def _build_hota_data(
    ground_truth: TrackingOutput,
    predictions: TrackingOutput,
) -> dict:
    """
    Build the data dict expected by trackeval.metrics.HOTA.eval_sequence().
    GT player identity is resolved via class_name (e.g. "White_14") mapped to a stable integer ID
    Predicted identity uses track_id (already integer).
    Predictions with track_id=None are excluded because they carry no identity.
    Parameters:
        - ground_truth: TrackingOutput from annotation files (source="ground_truth").
        - predictions: TrackingOutput from a tracker with track_id populated.
    Returns:
        Dict with keys: num_timesteps, num_gt_ids, num_tracker_ids, num_gt_dets,
        num_tracker_dets, gt_ids, tracker_ids, similarity_scores.
    """
    # Index predictions by frame
    pred_index = {frame.frame_index: frame.detections for frame in predictions.frames}

    # Integer ID maps fro HOTA compatibility and keep order stable
    gt_class_to_id: dict[str, int]   = {}
    pred_track_to_id: dict[int, int] = {}

    # Collect all unique IDs in a single pass to guarantee stable ordering
    for frame in ground_truth.frames:
        for detection in frame.detections:
            # Map GT class_name to integer ID if not seen before
            if detection.class_name not in gt_class_to_id:
                gt_class_to_id[detection.class_name] = len(gt_class_to_id)
    for frame in ground_truth.frames:
        for detection in pred_index.get(frame.frame_index, []):
            # Map predicted track_id to integer ID if not seen before; skip detections with track_id=None
            if detection.track_id is not None and detection.track_id not in pred_track_to_id:
                pred_track_to_id[detection.track_id] = len(pred_track_to_id)

    # Build lists of GT IDs, tracker IDs, and similarity scores (IoU) for each frame
    gt_ids_list: list[np.ndarray]      = []
    tracker_ids_list: list[np.ndarray] = []
    similarity_list: list[np.ndarray]  = []
    total_gt_dets      = 0
    total_tracker_dets = 0

    # Iterate over frames in ground truth
    for frame in ground_truth.frames:
        gt_detections = frame.detections

        # Only consider predicted detections with a valid track_id (identity) for matching; ignore others
        pred_detections = [detection for detection in pred_index.get(frame.frame_index, []) if detection.track_id is not None]

        # Map GT class_name and predicted track_id to integer IDs for this frame
        gt_ids = np.array([gt_class_to_id[detection.class_name]  for detection in gt_detections],   dtype=int)
        pred_ids = np.array([pred_track_to_id[detection.track_id]   for detection in pred_detections], dtype=int)

        # Compute IoU similarity matrix between GT and predicted detections for this frame; shape (N_gt, N_pred)
        if gt_detections and pred_detections:
            sim = _iou_matrix(gt_detections, pred_detections)   # trackeval expects IoU similarity (not distance)
        else:
            sim = np.zeros((len(gt_ids), len(pred_ids)), dtype=float)

        # Append results for this frame to the lists
        gt_ids_list.append(gt_ids)
        tracker_ids_list.append(pred_ids)
        similarity_list.append(sim)
        total_gt_dets      += len(gt_ids)
        total_tracker_dets += len(pred_ids)

    # Compile overall statistics and return the data dict expected by trackeval.metrics.HOTA.eval_sequence()
    return {
        "num_timesteps":     len(ground_truth.frames),
        "num_gt_ids":        len(gt_class_to_id),
        "num_tracker_ids":   len(pred_track_to_id),
        "num_gt_dets":       total_gt_dets,
        "num_tracker_dets":  total_tracker_dets,
        "gt_ids":            gt_ids_list,
        "tracker_ids":       tracker_ids_list,
        "similarity_scores": similarity_list,
    }


def compute_hota(
    ground_truth: TrackingOutput,
    predictions: TrackingOutput,
) -> HOTAMetrics:
    """Compute HOTA metrics between ground-truth annotations and tracker output using trackeval.
    Parameters:
        - ground_truth: TrackingOutput from annotation files (source="ground_truth").
        - predictions: TrackingOutput from a tracker with track_id populated.
    Returns:
        HOTAMetrics with scalar averages and per-alpha breakdowns.
    """
    data = _build_hota_data(ground_truth, predictions)
    res  = trackeval.metrics.HOTA().eval_sequence(data)

    return HOTAMetrics(
        hota = float(np.mean(res["HOTA"])),
        deta = float(np.mean(res["DetA"])),
        assa = float(np.mean(res["AssA"])),
        loca = float(np.mean(res["LocA"])),
        hota_per_alpha = res["HOTA"].tolist(),
        deta_per_alpha = res["DetA"].tolist(),
        assa_per_alpha = res["AssA"].tolist(),
        loca_per_alpha = res["LocA"].tolist(),
    )
