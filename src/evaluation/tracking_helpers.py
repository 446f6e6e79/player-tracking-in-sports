from collections.abc import Iterator

import numpy as np

# motmetrics uses np.asfarray which was removed in NumPy 2.0; patch it before the import
if not hasattr(np, "asfarray"):
    np.asfarray = lambda a, dtype=float: np.asarray(a, dtype=dtype)  # type: ignore[attr-defined]

import motmetrics as mm

from src.types.tracking import Detection, TrackingOutput, dets_to_xywh


def _iter_frame_pairs(
    ground_truth: TrackingOutput,
    pred_index: dict[int, list[Detection]],
) -> Iterator[tuple[int, list[Detection], list[Detection]]]:
    """Yield (frame_index, gt_detections, pred_detections) for each GT frame."""
    for frame in ground_truth.frames:
        gt_dets = frame.detections
        pred_dets = pred_index.get(frame.frame_index, [])
        yield frame.frame_index, gt_dets, pred_dets


def build_accumulator(
    ground_truth: TrackingOutput,
    predictions: TrackingOutput,
    iou_threshold: float,
) -> mm.MOTAccumulator:
    """Build a motmetrics accumulator over all GT-annotated frames.
    This accumulator is a basic data structure that stores information across the all frames
    It accumulates:
        - GT identities derived from class_name (e.g. "White_14") — the canonical identity in our annotation files.
        - Predicted identities from track_id.
        - IoU distance matrix for each frame, with pairs below the iou_threshold excluded via NaN.
    Parameters:
        - ground_truth: TrackingOutput from annotation files.
        - predictions: TrackingOutput from a tracker.
        - iou_threshold: minimum IoU for a match; pairs below this are excluded.
    Returns:
        Populated MOTAccumulator ready for metric computation.
    """
    # Index predictions by frame for efficient lookup
    pred_index = {frame.frame_index: frame.detections for frame in predictions.frames}

    # max_iou_dist is the IoU distance threshold
    max_iou_dist = 1.0 - iou_threshold

    # Stable integer ID map for GT class names, motmetrics requires numeric OIds
    gt_class_to_id: dict[str, int] = {}
    for frame in ground_truth.frames:
        for detection in frame.detections:
            if detection.class_name not in gt_class_to_id:
                gt_class_to_id[detection.class_name] = len(gt_class_to_id)

    # MOT Accumulator with auto_id=False since we manage IDs manually; this allows us to use class_name for GT identity and track_id for predicted identity
    acc = mm.MOTAccumulator(auto_id=False)

    for frame_index, gt_detections, pred_detections in _iter_frame_pairs(ground_truth, pred_index):
        # Build ID arrays for this frame based on GT class_name and predicted track_id
        gt_ids   = [gt_class_to_id[d.class_name] for d in gt_detections]
        pred_ids = [d.track_id                    for d in pred_detections]

        # mm.distances.iou_matrix expects XYWH boxes; convert from LTRB. Returns NaN for pairs above max_iou_dist.
        gt_boxes   = dets_to_xywh(gt_detections)
        pred_boxes = dets_to_xywh(pred_detections)
        dist = mm.distances.iou_matrix(gt_boxes, pred_boxes, max_iou=max_iou_dist)

        # Update the accumulator with this frame's GT IDs, predicted IDs, and distance matrix
        acc.update(gt_ids, pred_ids, dist, frameid=frame_index)

    return acc


def build_hota_data(
    ground_truth: TrackingOutput,
    predictions: TrackingOutput,
) -> dict:
    """
    Build the data dict expected by trackeval.metrics.HOTA.eval_sequence().
    GT player identity is resolved via class_name (e.g. "White_14") mapped to a stable integer ID.
    Predicted identity uses track_id (already integer).
    Parameters:
        - ground_truth: TrackingOutput from annotation files (source="ground_truth").
        - predictions: TrackingOutput from a tracker.
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
        for detection in pred_index.get(frame.frame_index, []):
            if detection.track_id not in pred_track_to_id:
                pred_track_to_id[detection.track_id] = len(pred_track_to_id)

    # Build lists of GT IDs, tracker IDs, and similarity scores (IoU) for each frame
    gt_ids_list: list[np.ndarray]      = []
    tracker_ids_list: list[np.ndarray] = []
    similarity_list: list[np.ndarray]  = []
    total_gt_dets      = 0
    total_tracker_dets = 0

    # Iterate over frames in ground truth
    for _, gt_detections, pred_detections in _iter_frame_pairs(ground_truth, pred_index):
        # Map GT class_name and predicted track_id to integer IDs for this frame
        gt_ids = np.array([gt_class_to_id[detection.class_name]  for detection in gt_detections],   dtype=int)
        pred_ids = np.array([pred_track_to_id[detection.track_id]   for detection in pred_detections], dtype=int)

        # Compute IoU similarity matrix between GT and predicted detections for this frame; shape (N_gt, N_pred)
        if gt_detections and pred_detections:
            # mm.distances.iou_matrix returns distance (1-IoU); invert to get similarity. trackeval expects IoU similarity (not distance)
            sim = 1.0 - mm.distances.iou_matrix(dets_to_xywh(gt_detections), dets_to_xywh(pred_detections), max_iou=1.0)
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
