import motmetrics as mm
import numpy as np

from src.evaluation.hota import compute_hota
from src.types.evaluation import (
    DetectionMetrics,
    EvaluationTrackingResult,
    IdentityMetrics,
)
from src.types.tracking import Detection, TrackingOutput


def _iou_distance_matrix(gt_detections: list[Detection], pred_detections: list[Detection], max_iou_dist: float) -> np.ndarray:
    """Compute (N_gt, N_pred) IoU distance matrix compatible with motmetrics.acc.update().
    This matrix encodes the pairwise IoU distance (1 - IoU) between GT and predicted detections for a single frame.
    Pairs with IoU below the threshold (distance above max_iou_dist) are set to NaN to signal motmetrics to ignore them for matching and metric computation.
    Parameters:
        - gt_detections: ground-truth detections for this frame.
        - pred_detections: predicted detections for this frame.
        - max_iou_dist: threshold above which pairs are excluded (= 1 - iou_threshold).
    Returns:
        (N_gt, N_pred) float array with distance values or NaN.
    """
    gt   = np.array([[d.bbox.x1, d.bbox.y1, d.bbox.x2, d.bbox.y2] for d in gt_detections],   dtype=float)
    pred = np.array([[d.bbox.x1, d.bbox.y1, d.bbox.x2, d.bbox.y2] for d in pred_detections], dtype=float)

    # Vectorised intersection: broadcast (N, 1, 4) and (1, M, 4)
    inter_x1 = np.maximum(gt[:, None, 0], pred[None, :, 0])
    inter_y1 = np.maximum(gt[:, None, 1], pred[None, :, 1])
    inter_x2 = np.minimum(gt[:, None, 2], pred[None, :, 2])
    inter_y2 = np.minimum(gt[:, None, 3], pred[None, :, 3])
    inter    = np.maximum(0.0, inter_x2 - inter_x1) * np.maximum(0.0, inter_y2 - inter_y1)

    # Union area is sum of individual areas minus intersection
    area_gt   = (gt[:, 2]   - gt[:, 0])   * (gt[:, 3]   - gt[:, 1])
    area_pred = (pred[:, 2] - pred[:, 0]) * (pred[:, 3] - pred[:, 1])
    union     = area_gt[:, None] + area_pred[None, :] - inter

    # IoU is intersection over union, defined to be 0 when union is zero (should not happen with valid boxes)
    iou  = np.where(union > 0, inter / union, 0.0)
    dist = 1.0 - iou

    # NaN signals motmetrics to not pair this (gt, pred) combination
    return np.where(dist > max_iou_dist, np.nan, dist)


def _build_accumulator(
    ground_truth: TrackingOutput,
    predictions: TrackingOutput,
    iou_threshold: float,
) -> mm.MOTAccumulator:
    """Build a motmetrics accumulator over all GT-annotated frames.
    This accumulator can be reused to compute both bbox-level detection metrics and ID-level metrics (IDF1 family).
    It accumulates:
        - GT identities derived from class_name (e.g. "White_14") since GT track_ids are None.
        - Predicted identities from track_id, excluding detections with track_id=None since they carry no identity.
        - IoU distance matrix for each frame, with pairs below the iou_threshold excluded via NaN.
    Parameters:
        - ground_truth: TrackingOutput from annotation files.
        - predictions: TrackingOutput from a tracker with track_id populated.
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

    # Map predicted track_id to integer ID if not seen before; skip detections with track_id=None
    for frame in ground_truth.frames:
        # First pass to build pred_track_to_id map to ensure stable ordering
        gt_detections   = frame.detections
        pred_detections = [d for d in pred_index.get(frame.frame_index, []) if d.track_id is not None]

        # Build ID arrays for this frame based on GT class_name and predicted track_id
        gt_ids   = [gt_class_to_id[d.class_name] for d in gt_detections]
        pred_ids = [d.track_id                    for d in pred_detections]

        # Compute IoU distance matrix for this frame; pairs with IoU below threshold will have NaN distance, signaling motmetrics to ignore them
        if gt_detections and pred_detections:
            dist = _iou_distance_matrix(gt_detections, pred_detections, max_iou_dist)
        else:
            # motmetrics accepts an empty matrix when one side is missing
            dist = np.empty((len(gt_ids), len(pred_ids)))

        # Update the accumulator with this frame's GT IDs, predicted IDs, and distance matrix
        acc.update(gt_ids, pred_ids, dist, frameid=frame.frame_index)

    return acc


def compute_detection_metrics(acc: mm.MOTAccumulator) -> DetectionMetrics:
    """Derive bbox-level detection metrics from a populated MOTAccumulator.
    Parameters:
        - acc: accumulator built by _build_accumulator().
    Returns:
        DetectionMetrics with tp, fp, fn, precision, recall, f1, mean_iou.
    """
    # motmetrics provides num_matches, num_false_positives, num_misses, precision, recall, and motp (mean IoU distance) directly; we compute f1 and mean_iou from these.
    mh = mm.metrics.create()

    # Compute detection metrics; motmetrics handles the matching based on the distance matrix and counts TP, FP, FN accordingly. Precision and recall are also computed, and we derive F1 and mean IoU from these.
    summary = mh.compute(
        acc,
        metrics=["num_matches", "num_false_positives", "num_misses", "precision", "recall", "motp"],
        name="eval",
    )

    # Extract values and compute derived metrics; convert to appropriate types
    tp  = int(summary["num_matches"]["eval"])
    fp  = int(summary["num_false_positives"]["eval"])
    fn  = int(summary["num_misses"]["eval"])
    precision   = float(summary["precision"]["eval"])
    recall   = float(summary["recall"]["eval"])
    f1  = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    # motp is mean IoU distance (1 - IoU) over matched pairs; invert to get mean IoU
    mean_iou = 1.0 - float(summary["motp"]["eval"])

    return DetectionMetrics(tp=tp, fp=fp, fn=fn, precision=precision, recall=recall, f1=f1, mean_iou=mean_iou)


def compute_identity_metrics(acc: mm.MOTAccumulator) -> IdentityMetrics:
    """Derive IDF1 family metrics from a populated MOTAccumulator.
    Parameters:
        - acc: accumulator built by _build_accumulator().
    Returns:
        IdentityMetrics with idtp, idfp, idfn, idp, idr, idf1.
    """
    mh = mm.metrics.create()
    summary = mh.compute(
        acc,
        metrics=["idtp", "idfp", "idfn", "idp", "idr", "idf1"],
        name="eval",
    )

    return IdentityMetrics(
        idtp  = int(summary["idtp"]["eval"]),
        idfp  = int(summary["idfp"]["eval"]),
        idfn  = int(summary["idfn"]["eval"]),
        idp   = float(summary["idp"]["eval"]),
        idr   = float(summary["idr"]["eval"]),
        idf1  = float(summary["idf1"]["eval"]),
    )


def evaluate_tracking(
    ground_truth: TrackingOutput,
    predictions: TrackingOutput,
    iou_threshold: float = 0.5,
) -> EvaluationTrackingResult:
    """Evaluate predicted tracking output against ground-truth annotations.
    Only frames present in ground_truth are evaluated (sparse GT-safe).
    GT player identity is resolved via class_name (e.g. "White_14"); predicted
    identity via track_id. track_id=None detections are excluded from identity metrics.
    Parameters:
        - ground_truth: TrackingOutput from annotation files (source="ground_truth").
        - predictions: TrackingOutput from a tracker with track_id populated.
        - iou_threshold: minimum IoU for a prediction to count as a true positive (default 0.5).
    Returns:
        EvaluationTrackingResult composed of DetectionMetrics, IdentityMetrics, HOTAMetrics.
    """
    # Build a shared accumulator once; all motmetrics-based compute_* functions reuse it
    acc = _build_accumulator(ground_truth, predictions, iou_threshold)

    return EvaluationTrackingResult(
        detection = compute_detection_metrics(acc),
        identity  = compute_identity_metrics(acc),
        hota      = compute_hota(ground_truth, predictions),
    )
