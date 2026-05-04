from collections.abc import Iterator

import numpy as np

# motmetrics uses np.asfarray which was removed in NumPy 2.0; patch it before the import
if not hasattr(np, "asfarray"):
    np.asfarray = lambda a, dtype=float: np.asarray(a, dtype=dtype)  # type: ignore[attr-defined]

import motmetrics as mm

from src.types.tracking import Detection, DetectionOutput, TrackingOutput, dets_to_xywh


def _validate_strictly_increasing(indices: list[int], label: str) -> None:
    for prev, curr in zip(indices, indices[1:]):
        if curr <= prev:
            raise ValueError(
                f"{label} frame indices must be strictly increasing. "
                f"Found non-increasing pair: {prev}, {curr}."
            )


def validate_evaluation_inputs(
    ground_truth: DetectionOutput | TrackingOutput,
    predictions: DetectionOutput | TrackingOutput,
    context: str,
) -> int:
    """Validate evaluation inputs and infer the prediction cadence stride.

    Returns:
        The stride between evaluated prediction frames and the underlying
        full-frame prediction sequence. A return value of 1 means the ground
        truth and predictions are aligned frame-for-frame. A value > 1 means the
        prediction sequence is denser than the annotated ground truth and should
        be sampled at that cadence.
    """
    if not ground_truth.frames:
        raise ValueError(f"{context}: ground-truth output has no frames.")
    if not predictions.frames:
        raise ValueError(f"{context}: prediction output has no frames.")

    if ground_truth.camera_id != predictions.camera_id:
        raise ValueError(
            f"{context}: camera mismatch: GT={ground_truth.camera_id}, "
            f"predictions={predictions.camera_id}."
        )

    gt_indices = [frame.frame_index for frame in ground_truth.frames]
    pred_indices = [frame.frame_index for frame in predictions.frames]

    _validate_strictly_increasing(gt_indices, f"{context} ground-truth")
    _validate_strictly_increasing(pred_indices, f"{context} predictions")

    if gt_indices[0] != 0:
        raise ValueError(
            f"{context}: ground-truth must be 0-based. "
            f"First frame_index is {gt_indices[0]}."
        )
    if pred_indices[0] != 0:
        raise ValueError(
            f"{context}: predictions must be 0-based. "
            f"First frame_index is {pred_indices[0]}."
        )

    if len(pred_indices) == len(gt_indices):
        if gt_indices != pred_indices:
            gt_set = set(gt_indices)
            pred_set = set(pred_indices)
            gt_only = sorted(gt_set - pred_set)[:5]
            pred_only = sorted(pred_set - gt_set)[:5]

            raise ValueError(
                f"{context}: frame index mismatch. "
                f"GT len/range={len(gt_indices)}/{gt_indices[0]}..{gt_indices[-1]}, "
                f"pred len/range={len(pred_indices)}/{pred_indices[0]}..{pred_indices[-1]}. "
                f"GT-only sample={gt_only}, pred-only sample={pred_only}."
            )
        return 1

    if len(pred_indices) % len(gt_indices) == 0:
        stride = len(pred_indices) // len(gt_indices)
        if stride <= 0:
            raise ValueError(f"{context}: invalid inferred prediction stride {stride}.")
        if gt_indices != list(range(len(gt_indices))):
            gt_set = set(gt_indices)
            pred_set = set(pred_indices)
            gt_only = sorted(gt_set - pred_set)[:5]
            pred_only = sorted(pred_set - gt_set)[:5]

            raise ValueError(
                f"{context}: ground-truth frame indices must be contiguous sample indices when predictions are denser. "
                f"GT len/range={len(gt_indices)}/{gt_indices[0]}..{gt_indices[-1]}, "
                f"pred len/range={len(pred_indices)}/{pred_indices[0]}..{pred_indices[-1]}, "
                f"inferred stride={stride}. GT-only sample={gt_only}, pred-only sample={pred_only}."
            )
        return stride

    gt_set = set(gt_indices)
    pred_set = set(pred_indices)
    gt_only = sorted(gt_set - pred_set)[:5]
    pred_only = sorted(pred_set - gt_set)[:5]

    raise ValueError(
        f"{context}: frame index mismatch. "
        f"GT len/range={len(gt_indices)}/{gt_indices[0]}..{gt_indices[-1]}, "
        f"pred len/range={len(pred_indices)}/{pred_indices[0]}..{pred_indices[-1]}. "
        f"GT-only sample={gt_only}, pred-only sample={pred_only}."
    )


def _iter_frame_pairs(
    ground_truth: TrackingOutput,
    pred_index: dict[int, list[Detection]],
    frame_stride: int,
) -> Iterator[tuple[int, list[Detection], list[Detection]]]:
    """Yield (frame_index, gt_detections, pred_detections) for each GT frame."""
    for gt_pos, frame in enumerate(ground_truth.frames):
        gt_dets = frame.detections
        pred_frame_index = gt_pos * frame_stride
        pred_dets = pred_index.get(pred_frame_index, [])
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
    frame_stride = validate_evaluation_inputs(ground_truth, predictions, context="Tracking evaluation")

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

    for frame_index, gt_detections, pred_detections in _iter_frame_pairs(ground_truth, pred_index, frame_stride):
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
    frame_stride = validate_evaluation_inputs(ground_truth, predictions, context="HOTA evaluation")

    # Index predictions by frame
    pred_index = {frame.frame_index: frame.detections for frame in predictions.frames}

    # Integer ID maps fro HOTA compatibility and keep order stable
    gt_class_to_id: dict[str, int]   = {}
    pred_track_to_id: dict[int, int] = {}

    # Collect all unique IDs in a single pass to guarantee stable ordering
    for gt_pos, frame in enumerate(ground_truth.frames):
        for detection in frame.detections:
            # Map GT class_name to integer ID if not seen before
            if detection.class_name not in gt_class_to_id:
                gt_class_to_id[detection.class_name] = len(gt_class_to_id)
        pred_frame_index = gt_pos * frame_stride
        for detection in pred_index.get(pred_frame_index, []):
            if detection.track_id not in pred_track_to_id:
                pred_track_to_id[detection.track_id] = len(pred_track_to_id)

    # Build lists of GT IDs, tracker IDs, and similarity scores (IoU) for each frame
    gt_ids_list: list[np.ndarray]      = []
    tracker_ids_list: list[np.ndarray] = []
    similarity_list: list[np.ndarray]  = []
    total_gt_dets      = 0
    total_tracker_dets = 0

    # Iterate over frames in ground truth
    for _, gt_detections, pred_detections in _iter_frame_pairs(ground_truth, pred_index, frame_stride):
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
