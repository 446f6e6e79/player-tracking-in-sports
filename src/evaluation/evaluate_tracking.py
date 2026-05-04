import numpy as np

import motmetrics as mm
import trackeval

from src.evaluation.helpers import (
    build_accumulator,
    build_hota_data,
)
from src.types.evaluation import (
    TrackingMetrics,
    HOTAMetrics,
    IdentityMetrics,
)
from src.types.tracking import TrackingOutput
from src.evaluation.evaluate_detection import compute_detection_metrics

# Create a metrics handler
_MH = mm.metrics.create()



def compute_identity_metrics(acc: mm.MOTAccumulator) -> IdentityMetrics:
    """Derive IDF1 family metrics from a populated MOTAccumulator.
    - True positives (tp): number of GT detections correctly identified (IoU > threshold with a prediction of the same identity).
    - False positives (fp): number of predicted detections that are IoU > threshold with a GT detection of a different identity, or with no GT match.
    - False negatives (fn): number of GT detections that are IoU > threshold with a predicted detection of a different identity, or with no prediction match.
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
    frame_stride: int = 5,
) -> HOTAMetrics:
    """Compute HOTA metrics between ground-truth annotations and tracker output using trackeval.
    Parameters:
        - ground_truth: TrackingOutput from annotation files (source="ground_truth").
        - predictions: TrackingOutput from a tracker.
    Returns:
        HOTAMetrics with scalar averages and per-alpha breakdowns.
            - hota: representing the overall HOTA score (aggregate geometric mean of DetA and AssA averaged over alpha thresholds)
            - deta: representing the detection accuracy (mean IoU of matched pairs averaged over alpha thresholds)
            - assa: representing the association accuracy (F1 score of matched pairs averaged over alpha thresholds)
            - loca: representing the localisation accuracy (mean IoU of matched pairs averaged over alpha thresholds)
            - hota_per_alpha, deta_per_alpha, assa_per_alpha, loca_per_alpha: lists of values at each alpha threshold
    Note: trackeval.metrics.HOTA.eval_sequence() computes HOTA at multiple alpha thresholds (e.g. 0.5, 0.55, ..., 0.95) and returns the average as well as the per-alpha values.
    """
    data = build_hota_data(ground_truth, predictions, frame_stride)
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
    frame_stride: int = 5,
) -> TrackingMetrics:
    """Evaluate predicted tracking output against ground-truth annotations.
    Parameters:
        - ground_truth: TrackingOutput from annotation files (source="ground_truth").
        - predictions: TrackingOutput from a tracker.
        - iou_threshold: minimum IoU for a prediction to count as a true positive (default 0.5).
        - frame_stride: step between GT annotation frames and prediction frame indices (default 5).
    Returns:
        TrackingMetrics composed of DetectionMetrics, IdentityMetrics, HOTAMetrics.
    """
    # Build a shared accumulator once; all motmetrics-based compute_* functions reuse it
    acc = build_accumulator(ground_truth, predictions, iou_threshold, frame_stride)

    return TrackingMetrics(
        identity  = compute_identity_metrics(acc),
        hota      = compute_hota(ground_truth, predictions, frame_stride),
    )
