from collections import defaultdict

import numpy as np

from src.calibration.camera_data import CameraData
from src.types.geometry import TriangulationOutput, RectifiedPointsOutput
from src.types.evaluation import (
    GeometryMetrics,
    ReprojectionMetrics,
    TrajectoryMetrics,
)
from src.geometry.triangulation import project_points


def compute_reprojection_metrics(
    triangulations: TriangulationOutput,
    cam: CameraData,
    tracking_output: RectifiedPointsOutput,
    frame_stride: int = 5,
) -> ReprojectionMetrics:
    """
    Project GT 3D points into cam's pixel space and compare against annotated
    2D positions. This should be computed on the same tracking output used to build the triangulations and the frame indices should match those in the triangulations.
    Parameters:
        - triangulations: TriangulationOutput driving the frame clock.
        - cam: target camera for projection.
        - tracking_output: per-camera rectified annotated outputs.
    Returns:
        ReprojectionMetrics with error statistics and accuracy@5/10/20px.
    """
    # Build an index of annotated points by frame for easy lookup.
    annotated_index = {f.frame_index: f.points for f in tracking_output.frames}
    # Compute reprojection error for each matched pair of 3D point and annotated 2D point.
    all_errors: list[float] = []

    for gt_pos, frame in enumerate(triangulations.frames):
        if not frame.points:
            continue
        annotated_by_class = {d.class_name: d for d in annotated_index.get(gt_pos * frame_stride, [])}

        world_pts = np.array([[p.x, p.y, p.z] for p in frame.points], dtype=np.float32)
        pixels    = project_points(world_pts, cam)

        # For each projected point, if there is a corresponding annotated point with the same class_name
        for i, pt in enumerate(frame.points):
            pred = annotated_by_class.get(pt.class_name)
            if pred is not None:
                px, py = float(pixels[i, 0]), float(pixels[i, 1])
                all_errors.append(float(np.sqrt((px - pred.x) ** 2 + (py - pred.y) ** 2)))

    # Compute error statistics and accuracy at different pixel thresholds.
    errors     = np.array(all_errors, dtype=np.float64)
    return ReprojectionMetrics(
        mean_error_px    = float(np.mean(errors)),
        median_error_px  = float(np.median(errors)),
        std_error_px     = float(np.std(errors)),
        rmse_px          = float(np.sqrt(np.mean(errors ** 2))),
        accuracy_at_5px  = float(np.mean(errors < 5.0)),
        accuracy_at_10px = float(np.mean(errors < 10.0)),
        accuracy_at_20px = float(np.mean(errors < 20.0)),
    )


def compute_trajectory_metrics(
    triangulations: TriangulationOutput,
    cam: CameraData,
    tracking_output: RectifiedPointsOutput,
    frame_stride: int = 5,
) -> TrajectoryMetrics:
    """
    Build per-identity GT and pred pixel trajectories, then compute ADE, FDE,
    MTE, smoothness and jitter. Identities matched by class_name.

    Parameters:
        - triangulations: TriangulationOutput driving the frame clock.
        - cam: target camera for projection.
        - tracking_output: per-camera rectified annotated outputs.
    Returns:
        TrajectoryMetrics aggregated over all matched class_name trajectory pairs.
    """
    # Build an index of annotated points by frame for easy lookup.
    annotated_index = {f.frame_index: f.points for f in tracking_output.frames}

    # Build trajectory maps for triangulated and ground truth points.
    triangulated_traj:   dict[str, list[tuple[int, float, float]]] = defaultdict(list)
    gt_traj: dict[str, list[tuple[int, float, float]]] = defaultdict(list)

    # Iterate through triangulated frames and project points into pixel space, building trajectories keyed by class_name.
    for gt_pos, frame in enumerate(triangulations.frames):
        if frame.points:
            world_pts = np.array([[p.x, p.y, p.z] for p in frame.points], dtype=np.float32)
            pixels    = project_points(world_pts, cam)
            for i, pt in enumerate(frame.points):
                triangulated_traj[pt.class_name].append((gt_pos, float(pixels[i, 0]), float(pixels[i, 1])))
        for d in annotated_index.get(gt_pos * frame_stride, []):
            gt_traj[d.class_name].append((gt_pos, float(d.x), float(d.y)))

    # For each identity (class_name) in GT, if it exists in the triangulated trajectories, compute ADE, FDE, MTE, smoothness and jitter.
    ade_values:        list[float] = []
    fde_values:        list[float] = []
    mte_values:        list[float] = []
    total_trajectories   = 0
    trajectory_fragments = 0
    for gt_class, gt_pts in gt_traj.items():
        if gt_class not in triangulated_traj:
            continue
        
        # Match GT and predicted trajectories by class_name. 
        pred_pts     = triangulated_traj[gt_class]
        gt_pos_map   = {fp: (x, y) for fp, x, y in gt_pts}
        pred_pos_map = {fp: (x, y) for fp, x, y in pred_pts}

        # Find common frames between GT and predicted trajectories to compute errors. 
        common_frames = sorted(gt_pos_map.keys() & pred_pos_map.keys())
        if not common_frames:
            continue
        
        # Compute metrics for the matched trajectory.
        total_trajectories += 1
        errors = np.array([
            np.linalg.norm(np.array(gt_pos_map[f]) - np.array(pred_pos_map[f]))
            for f in common_frames
        ], dtype=np.float64)
        ade_values.append(float(np.mean(errors)))
        fde_values.append(float(errors[-1]))
        mte_values.append(float(np.median(errors)))

        if len(common_frames) >= 2:
            trajectory_fragments += sum(
                1 for a, b in zip(common_frames, common_frames[1:]) if b - a > 1
            )


    return TrajectoryMetrics(
        ade_px               = float(np.mean(ade_values)),
        fde_px               = float(np.mean(fde_values)),
        mte_px               = float(np.mean(mte_values)),
        total_trajectories   = total_trajectories,
        trajectory_fragments = trajectory_fragments,
    )

def evaluate_geometry(
    triangulations: TriangulationOutput,
    annotated_tracking: dict[str, RectifiedPointsOutput],
    frame_stride: int = 5,
) -> dict[str, GeometryMetrics]:
    """
    Evaluate 3D tracking quality by projecting the annotated GT
    `TriangulationOutput` into each camera's 2D pixel space and comparing
    against the per-camera rectified annotated outputs.

    For each camera the function computes:
      - ReprojectionMetrics  (mean/median/RMSE error, accuracy@5/10/20px)
      - TrajectoryMetrics    (ADE, FDE, MTE, smoothness, jitter,
                              trajectory count and fragment count)

    Parameters:
        - triangulations: annotated `TriangulationOutput` (source of GT
          identity and 3D world positions). `Point3D.class_name` is the GT
          identity key (e.g. "White_14").
        - annotated_tracking: map of camera_id → rectified annotated outputs
          to evaluate against. Frame indices must match those in triangulations.
    Returns:
        Dict mapping camera_id → GeometryMetrics (reprojection + trajectory).
    """
    results: dict[str, GeometryMetrics] = {}
    # Iterate over cameras for which we have annotated GT tracking outputs, compute reprojection and trajectory metrics for each
    for camera_id, tracking_output in sorted(annotated_tracking.items()):
        cam = CameraData.load(camera_id)
        results[camera_id] = GeometryMetrics(
            reprojection = compute_reprojection_metrics(triangulations, cam, tracking_output, frame_stride),
            trajectory   = compute_trajectory_metrics(triangulations, cam, tracking_output, frame_stride),
        )

    return results
