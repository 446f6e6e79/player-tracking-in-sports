from dataclasses import dataclass, field


@dataclass
class DetectionMetrics:
    """Bbox-level detection quality, IoU-matched per frame."""
    tp: int             # true positives: number of GT detections with IoU > 0.5 to a prediction
    fp: int             # false positives: number of predicted detections with no GT match (IoU ≤ 0.5 to all GT)
    fn: int             # false negatives: number of GT detections with IoU ≤ 0.5 to all predictions
    precision: float    # tp / (tp + fp)
    recall: float       # tp / (tp + fn)
    f1: float           # harmonic mean of precision and recall
    mean_iou: float     # average IoU over matched pairs


@dataclass
class IdentityMetrics:
    """ID-level metrics (IDF1 family) over the whole sequence.
    GT player identity comes from class_name (e.g. "White_14"); predicted identity from track_id.
    """
    idtp: int       # ID true positives: number of GT detections correctly identified (IoU > 0.5 with a prediction of the same identity)
    idfp: int       # ID false positives: number of predicted detections that are IoU > 0.5 with a GT detection of a different identity, or with no GT match
    idfn: int       # ID false negatives: number of GT detections that are IoU > 0.5 with a predicted detection of a different identity, or with no prediction match
    idp: float      # ID precision
    idr: float      # ID recall
    idf1: float     # harmonic mean of idp and idr

@dataclass
class HOTAMetrics:
    """Higher Order Tracking Accuracy (Luiten et al., IJCV 2021)."""
    hota: float                                                 # Aggregate Geometric mean of Detection Accuracy and Association Accuracy averaged over alpha thresholds
    deta: float                                                 # Detection Accuracy (mean IoU of matched pairs averaged over alpha thresholds)
    assa: float                                                 # Association Accuracy (F1 score of matched pairs averaged over alpha thresholds)
    loca: float                                                 # Localisation Accuracy (mean IoU of matched pairs averaged over alpha thresholds)
    hota_per_alpha: list[float] = field(default_factory=list)   # Values of HOTA at each alpha threshold
    deta_per_alpha: list[float] = field(default_factory=list)   # Values of DetA at each alpha threshold
    assa_per_alpha: list[float] = field(default_factory=list)   # Values of AssA at each alpha threshold
    loca_per_alpha: list[float] = field(default_factory=list)   # Values of LocA at each alpha threshold


@dataclass
class EvaluationTrackingResult:
    """Top-level container returned by evaluate_tracking().
    Each field is a self-contained metric family that can be inspected independently.
    """
    detection: DetectionMetrics # Bbox-level detection quality, IoU-matched per frame
    identity: IdentityMetrics   # ID-level metrics (IDF1 family) over the whole sequence
    hota: HOTAMetrics           # Higher Order Tracking Accuracy (Luiten et al., IJCV 2021) with scalar averages and per-alpha breakdowns
