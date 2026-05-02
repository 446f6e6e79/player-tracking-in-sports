import numpy as np

import motmetrics as mm
import trackeval

from src.evaluation.tracking_helpers import build_accumulator, build_hota_data
from src.types.evaluation import (
    DetectionMetrics,
    EvaluationTrackingResult,
    HOTAMetrics,
    IdentityMetrics,
)
from src.types.tracking import TrackingOutput

# Create a metrics handler
_MH = mm.metrics.create()


def compute_detection_metrics(acc: mm.MOTAccumulator) -> DetectionMetrics:
    """Derive bbox-level detection metrics from a populated MOTAccumulator.
    - True positives (tp): number of predicted boxes that correctly match a ground-truth box (IoU > threshold).
    - False positives (fp): number of predicted boxes that do not match any ground-truth box (noise and extra detections).
    - False negatives (fn): number of ground-truth boxes that do not match any predicted
    - Mean IoU: average IoU over matched pairs, reflecting localization quality of true positives.
    Parameters:
        - acc: accumulator built by _build_accumulator().
    Returns:
        DetectionMetrics with tp, fp, fn, precision, recall, f1, mean_iou.
    """
    # Compute detection metrics; motmetrics handles the matching based on the distance matrix and counts TP, FP, FN accordingly. Precision and recall are also computed, and we derive F1 and mean IoU from these.
    summary = _MH.compute(
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
    - True positives (tp): number of GT detections correctly identified (IoU > threshold with a prediction of the same identity).
    - False positives (fp): number of predicted detections that are IoU > threshold with a GT detection of a different identity, or with no GT match (including track_id=None detections).
    - False negatives (fn): number of GT detections that are IoU > threshold with a predicted detection of a different identity, or with no prediction match (including track_id=None detections).
    Parameters:
        - acc: accumulator built by build_accumulator().
    Returns:
        IdentityMetrics with tp, fp, fn, precision, recall, f1, idsw.
    """
    # Compute ID-level metrics
    summary = _MH.compute(
        acc,
        metrics=["idtp", "idfp", "idfn", "idp", "idr", "idf1"],
        name="eval",
    )

    # Extract values and compute derived metrics; convert to appropriate types
    return IdentityMetrics(
        tp = int(summary["idtp"]["eval"]),
        fp = int(summary["idfp"]["eval"]),
        fn = int(summary["idfn"]["eval"]),
        precision = float(summary["idp"]["eval"]),
        recall   = float(summary["idr"]["eval"]),
        f1  = float(summary["idf1"]["eval"]),
    )


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
            - hota: representing the overall HOTA score (aggregate geometric mean of DetA and AssA averaged over alpha thresholds)
            - deta: representing the detection accuracy (mean IoU of matched pairs averaged over alpha thresholds)
            - assa: representing the association accuracy (F1 score of matched pairs averaged over alpha thresholds)
            - loca: representing the localisation accuracy (mean IoU of matched pairs averaged over alpha thresholds)
            - hota_per_alpha, deta_per_alpha, assa_per_alpha, loca_per_alpha: lists of values at each alpha threshold
    Note: trackeval.metrics.HOTA.eval_sequence() computes HOTA at multiple alpha thresholds (e.g. 0.5, 0.55, ..., 0.95) and returns the average as well as the per-alpha values.
    """
    data = build_hota_data(ground_truth, predictions)
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


def evaluate_tracking(
    ground_truth: TrackingOutput,
    predictions: TrackingOutput,
    iou_threshold: float = 0.5,
) -> EvaluationTrackingResult:
    """Evaluate predicted tracking output against ground-truth annotations.
    Parameters:
        - ground_truth: TrackingOutput from annotation files (source="ground_truth").
        - predictions: TrackingOutput from a tracker with track_id populated.
        - iou_threshold: minimum IoU for a prediction to count as a true positive (default 0.5).
    Returns:
        EvaluationTrackingResult composed of DetectionMetrics, IdentityMetrics, HOTAMetrics.
    """
    # Build a shared accumulator once; all motmetrics-based compute_* functions reuse it
    acc = build_accumulator(ground_truth, predictions, iou_threshold)

    return EvaluationTrackingResult(
        detection = compute_detection_metrics(acc),
        identity  = compute_identity_metrics(acc),
        hota      = compute_hota(ground_truth, predictions),
    )
