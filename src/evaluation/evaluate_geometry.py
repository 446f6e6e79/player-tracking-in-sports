from collections import defaultdict

import numpy as np

from src.calibration.camera_data import CameraData
from src.types.geometry import TriangulationOutput
from src.types.tracking import TrackingOutput, TrackedDetection
from src.types.evaluation import (
    GeometryMetrics,
    ReprojectionMetrics,
    TrajectoryMetrics,
)
from src.evaluation.geometry_helpers import (
    _ProjectedPoint,
    project_triangulation,
    build_annotated_index,
)


# ---------------------------------------------------------------------------
# Reprojection metrics
# ---------------------------------------------------------------------------

def compute_reprojection_metrics(
    triangulation: TriangulationOutput,
    projected_index: dict[int, list[_ProjectedPoint]],
    annotated_index: dict[int, list[TrackedDetection]],
    frame_stride: int,
) -> tuple[ReprojectionMetrics]:
    """
    Aggregate frame-level identity matches into ReprojectionMetrics.
    Also returns the full list of matches for downstream trajectory analysis.

    Parameters:
        - triangulation: annotated TriangulationOutput (drives the frame clock).
        - projected_index: frame_index → projected triangulation points (from project_triangulation).
        - annotated_index: frame_index → annotated detections (from build_annotated_index).
        - frame_stride: positional stride between triangulation and prediction frames.
    Returns:
        ReprojectionMetrics, and the flat list of all frame-level _FrameMatch pairs.
    """
    all_errors: list[float] = []
    total_matches = 0
    unmatched_gt = 0
    unmatched_predictions = 0

    # Iterate through triangulation frames
    for gt_pos, frame in enumerate(triangulation.frames):
        # map triangulation position to prediction frame index
        pred_frame_idx = gt_pos * frame_stride

        projected = projected_index.get(frame.frame_index, []) or projected_index.get(pred_frame_idx, [])
        annotated = annotated_index.get(frame.frame_index, []) or annotated_index.get(pred_frame_idx, [])

        # Build lookup tables by class_name
        projected_by_class: dict[str, _ProjectedPoint] = {}
        annotated_by_class: dict[str, TrackedDetection] = {}

        for p in projected:
            projected_by_class[p.class_name] = p
        for d in annotated:
            annotated_by_class[d.class_name] = d

        # Match GT to predictions by class_name
        matched_predictions = set()
        for class_name, gt_point in projected_by_class.items():
            if class_name in annotated_by_class:
                pred_det = annotated_by_class[class_name]
                matched_predictions.add(class_name)
                
                # Compute reprojection error
                cx, cy = pred_det.bbox.get_center()
                error = np.sqrt((gt_point.x - cx) ** 2 + (gt_point.y - cy) ** 2)
                all_errors.append(float(error))
                total_matches += 1
            else:
                unmatched_gt += 1

        # Count unmatched predictions
        for class_name in annotated_by_class:
            if class_name not in matched_predictions:
                unmatched_predictions += 1

    if all_errors:
        errors     = np.array(all_errors, dtype=np.float64)
        mean_err   = float(np.mean(errors))
        median_err = float(np.median(errors))
        std_err    = float(np.std(errors))
        rmse       = float(np.sqrt(np.mean(errors ** 2)))
        acc_2px    = float(np.mean(errors < 2.0))
        acc_5px    = float(np.mean(errors < 5.0))
        acc_10px   = float(np.mean(errors < 10.0))
    else:
        mean_err = median_err = std_err = rmse = 0.0
        acc_2px  = acc_5px = acc_10px = 0.0

    return ReprojectionMetrics(
        mean_error_px         = mean_err,
        median_error_px       = median_err,
        std_error_px          = std_err,
        rmse_px               = rmse,
        accuracy_at_2px       = acc_2px,
        accuracy_at_5px       = acc_5px,
        accuracy_at_10px      = acc_10px,
        total_matches         = total_matches,
        unmatched_predictions = unmatched_predictions,
        unmatched_gt          = unmatched_gt,
    )


# ---------------------------------------------------------------------------
# Trajectory metrics
# ---------------------------------------------------------------------------

def compute_trajectory_metrics(
    triangulation: TriangulationOutput,
    projected_index: dict[int, list[_ProjectedPoint]],
    annotated_index: dict[int, list[TrackedDetection]],
    frame_stride: int,
) -> TrajectoryMetrics:
    """
    Build per-identity GT and pred pixel trajectories, then compute ADE, FDE,
    MTE, smoothness and jitter. Since identities are already aligned, no
    association step is needed — GT class_name maps directly to pred class_name.

    Parameters:
        - triangulation: annotated TriangulationOutput (drives the frame clock).
        - projected_index: frame_index → projected triangulation points.
        - annotated_index: frame_index → annotated detections.
        - frame_stride: positional stride between triangulation and prediction frames.
    Returns:
        TrajectoryMetrics aggregated over all GT↔pred trajectory pairs.
    """
    # Build GT and pred trajectories in a single pass over frames
    gt_traj:   dict[str, list[tuple[int, float, float]]] = defaultdict(list)
    pred_traj: dict[str, list[tuple[int, float, float]]] = defaultdict(list)

    # Iterate triangulation frames and look up projected/annotated detections
    # by the triangulation frame index; apply `frame_stride` via position
    # if predictions are sampled at a different rate.
    for gt_pos, frame in enumerate(triangulation.frames):
        # map triangulation position to prediction frame index
        pred_frame_idx = gt_pos * frame_stride

        projected = projected_index.get(frame.frame_index, []) or projected_index.get(pred_frame_idx, [])
        annotated = annotated_index.get(frame.frame_index, []) or annotated_index.get(pred_frame_idx, [])

        for p in projected:
            gt_traj[p.class_name].append((gt_pos, p.x, p.y))
        for d in annotated:
            cx, cy = d.bbox.get_center()
            pred_traj[d.class_name].append((gt_pos, float(cx), float(cy)))

    # Match GT and pred by class_name
    ade_values:        list[float] = []
    fde_values:        list[float] = []
    mte_values:        list[float] = []
    smoothness_values: list[float] = []
    jitter_values:     list[float] = []
    total_trajectories   = 0
    trajectory_fragments = 0

    for gt_class, gt_pts in gt_traj.items():
        # Match by class_name directly
        if gt_class not in pred_traj:
            continue

        pred_pts = pred_traj[gt_class]

        gt_pos_map:   dict[int, tuple[float, float]] = {fp: (x, y) for fp, x, y in gt_pts}
        pred_pos_map: dict[int, tuple[float, float]] = {fp: (x, y) for fp, x, y in pred_pts}

        common_frames = sorted(gt_pos_map.keys() & pred_pos_map.keys())
        if not common_frames:
            continue

        total_trajectories += 1

        errors = np.array([
            np.linalg.norm(np.array(gt_pos_map[f]) - np.array(pred_pos_map[f]))
            for f in common_frames
        ], dtype=np.float64)

        ade_values.append(float(np.mean(errors)))
        fde_values.append(float(errors[-1]))
        mte_values.append(float(np.median(errors)))

        pred_pts_sorted = sorted(pred_pts)
        if len(pred_pts_sorted) >= 2:
            pred_arr      = np.array([(x, y) for _, x, y in pred_pts_sorted], dtype=np.float64)
            displacements = np.linalg.norm(np.diff(pred_arr, axis=0), axis=1)
            smoothness_values.append(
                float(np.mean(np.abs(np.diff(displacements)))) if len(displacements) > 1 else 0.0
            )
            jitter_values.append(float(np.std(displacements)))

        covered_sorted = sorted(gt_pos_map.keys() & pred_pos_map.keys())
        if len(covered_sorted) >= 2:
            trajectory_fragments += sum(
                1 for a, b in zip(covered_sorted, covered_sorted[1:]) if b - a > 1
            )

    def _safe_mean(lst: list[float]) -> float:
        return float(np.mean(lst)) if lst else 0.0

    return TrajectoryMetrics(
        ade_px                   = _safe_mean(ade_values),
        fde_px                   = _safe_mean(fde_values),
        mte_px                   = _safe_mean(mte_values),
        trajectory_smoothness_px = _safe_mean(smoothness_values),
        jitter_px                = _safe_mean(jitter_values),
        total_trajectories       = total_trajectories,
        trajectory_fragments     = trajectory_fragments,
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def evaluate_geometry(
    triangulations: TriangulationOutput,
    annotated_tracking: dict[str, TrackingOutput],
    frame_stride: int = 1,
) -> dict[str, GeometryMetrics]:
    """
    Evaluate 3D tracking quality by projecting the annotated GT
    `TriangulationOutput` into each camera's 2D pixel space and comparing
    against the per-camera annotated `TrackingOutput`.

    Identities are matched directly by class_name ↔ track_id — no spatial
    thresholding is applied. The reprojection error for each matched pair is
    computed purely as a quality measurement.

    For each camera the function computes:
      - ReprojectionMetrics  (mean/median/RMSE error, accuracy@Npx,
                              matched/unmatched counts)
      - TrajectoryMetrics    (ADE, FDE, MTE, smoothness, jitter,
                              trajectory count and fragment count)

    Parameters:
        - triangulations: annotated `TriangulationOutput` (source of GT
          identity and 3D world positions). `Point3D.class_name` is the GT
          identity key (e.g. "White_14").
        - annotated_tracking: map of camera_id → annotated `TrackingOutput`
          to evaluate against.
        - frame_stride: positional stride between triangulation frames and
          prediction frame indices (default 1, i.e. no stride).
    Returns:
        Dict mapping camera_id → GeometryMetrics (reprojection + trajectory).
    """
    results: dict[str, GeometryMetrics] = {}

    for camera_id, tracking_output in sorted(annotated_tracking.items()):
        cam = CameraData.load(camera_id)

        projected_index = project_triangulation(triangulations, cam)
        annotated_index = build_annotated_index(tracking_output)

        reprojection = compute_reprojection_metrics(
            triangulations, projected_index, annotated_index, frame_stride
        )
        trajectory = compute_trajectory_metrics(
            triangulations, projected_index, annotated_index, frame_stride
        )

        results[camera_id] = GeometryMetrics(
            reprojection = reprojection,
            trajectory   = trajectory,
        )

    return results