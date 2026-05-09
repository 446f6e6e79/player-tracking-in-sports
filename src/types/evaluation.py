from dataclasses import dataclass, field

# --- Tracking evaluation metrics for a single camera view. ---

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
    tp: int                 # ID true positives: number of GT detections correctly identified (IoU > 0.5 with a prediction of the same identity)
    fp: int                 # ID false positives: number of predicted detections that are IoU > 0.5 with a GT detection of a different identity, or with no GT match
    fn: int                 # ID false negatives: number of GT detections that are IoU > 0.5 with a predicted detection of a different identity, or with no prediction match
    precision: float        # ID precision
    recall: float           # ID recall
    f1: float               # harmonic mean of idp and idr

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
class TrackingMetrics:
    """Top-level container returned by evaluate_tracking().
    Each field is a self-contained metric family that can be inspected independently.
    """
    identity: IdentityMetrics   # ID-level metrics (IDF1 family) over the whole sequence
    hota: HOTAMetrics           # Higher Order Tracking Accuracy (Luiten et al., IJCV 2021) with scalar averages and per-alpha breakdowns


# --- Geometry evaluation metrics for a single camera view. ---

@dataclass
class ReprojectionMetrics:
    """
    Standard reprojection-quality metrics computed in image space.
    Predicted 3D points are projected into the camera image plane and
    compared against GT annotated 2D positions.
    """
    mean_error_px: float              # Mean reprojection error (pixels)
    median_error_px: float            # Median reprojection error (pixels)
    std_error_px: float               # Standard deviation of reprojection error
    rmse_px: float                    # Root Mean Squared reprojection error

    accuracy_at_2px: float            # Percentage of projections with error < 2 px
    accuracy_at_5px: float            # Percentage of projections with error < 5 px
    accuracy_at_10px: float           # Percentage of projections with error < 10 px

    total_matches: int                # Number of matched GT/prediction pairs
    unmatched_predictions: int        # Predicted points without GT match
    unmatched_gt: int                 # GT points without prediction match


@dataclass
class TrajectoryMetrics:
    """
    Temporal trajectory evaluation metrics.

    Computed on trajectories over time after associating predicted tracks
    with GT trajectories.
    """
    ade_px: float                     # Average Displacement Error
    fde_px: float                     # Final Displacement Error
    mte_px: float                     # Median Trajectory Error

    trajectory_smoothness_px: float   # Average frame-to-frame displacement variation
    jitter_px: float                  # High-frequency trajectory instability

    total_trajectories: int           # Number of evaluated trajectories
    trajectory_fragments: int         # Number of interrupted trajectories


@dataclass
class GeometryMetrics:
    """
    Full set of SOTA geometry evaluation metrics for a single camera view, returned by evaluate_geometry().

    """
    reprojection: ReprojectionMetrics   # Standard reprojection-quality metrics computed in image space
    trajectory: TrajectoryMetrics       # Temporal trajectory evaluation metrics computed on trajectories over time after associating predicted tracks with GT trajectories


