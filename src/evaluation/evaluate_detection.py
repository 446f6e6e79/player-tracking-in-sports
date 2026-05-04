import motmetrics as mm

from src.evaluation.helpers import build_accumulator
from src.types.evaluation import DetectionMetrics
from src.types.tracking import DetectionOutput

_MH = mm.metrics.create()

def compute_detection_metrics(acc: mm.MOTAccumulator) -> DetectionMetrics:
    """Derive bbox-level detection metrics from a populated MOTAccumulator.
    - True positives (tp): number of predicted boxes that correctly match a ground-truth box (IoU > threshold).
    - False positives (fp): number of predicted boxes that do not match any ground-truth box (noise and extra detections).
    - False negatives (fn): number of ground-truth boxes that do not match any predicted
    - Mean IoU: average IoU over matched pairs, reflecting localization quality of true positives.
    Parameters:
        - acc: accumulator built by build_accumulator().
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



def evaluate_detection(
    ground_truth: DetectionOutput,
    predictions: DetectionOutput,
    iou_threshold: float = 0.5,
    frame_stride: int = 5,
) -> DetectionMetrics:
    """Evaluate predicted detections against ground-truth annotations.

    Suitable for detection-only models (e.g. MOG2). Identity is not evaluated —
    only bbox-level quality.

    Parameters:
        - ground_truth: annotations (DetectionOutput).
        - predictions: detector output (DetectionOutput).
        - iou_threshold: minimum IoU for a prediction to count as a true positive (default 0.5).
        - frame_stride: step between GT annotation frames and prediction frame indices (default 5).
    Returns:
        DetectionMetrics with tp, fp, fn, precision, recall, f1, mean_iou.
    """
    # build_accumulator auto-detects use_identity=False for DetectionOutput
    acc = build_accumulator(ground_truth, predictions, iou_threshold, frame_stride)

    return compute_detection_metrics(acc)