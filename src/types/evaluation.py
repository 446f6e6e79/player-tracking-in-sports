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
    compared against GT annotated 2D positions. The image-plane pixel error is
    scaled to a physical distance (mm/px = Z / f at the point's camera-frame
    depth Z), so all values are reported in millimeters.
    """
    mean_error_mm: float              # Mean reprojection error (mm)
    median_error_mm: float            # Median reprojection error (mm)
    std_error_mm: float               # Standard deviation of reprojection error (mm)
    rmse_mm: float                    # Root Mean Squared reprojection error (mm)

    accuracy_at_25mm: float           # Fraction of projections with error < 25 mm
    accuracy_at_50mm: float           # Fraction of projections with error < 50 mm
    accuracy_at_100mm: float          # Fraction of projections with error < 100 mm

@dataclass
class TrajectoryMetrics:
    """
    Temporal trajectory evaluation metrics.

    Computed on trajectories over time after associating predicted tracks
    with GT trajectories. The image-plane displacement is scaled to a physical
    distance (mm/px = Z / f at the predicted point's depth), so values are in mm.
    """
    ade_mm: float                     # Average Displacement Error (mm)
    fde_mm: float                     # Final Displacement Error (mm)
    mte_mm: float                     # Median Trajectory Error (mm)

    total_trajectories: int           # Number of evaluated trajectories
    trajectory_fragments: int         # Number of interrupted trajectories


@dataclass
class GeometryMetrics:
    """Full set of SOTA geometry evaluation metrics for a single camera view, returned by evaluate_geometry()."""
    reprojection: ReprojectionMetrics   # Standard reprojection-quality metrics computed in image space
    trajectory: TrajectoryMetrics       # Temporal trajectory evaluation metrics computed on trajectories over time after associating predicted tracks with GT trajectories


