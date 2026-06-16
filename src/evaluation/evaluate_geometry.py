from collections import defaultdict
from typing import Iterable

import numpy as np

from src.calibration.camera_data import CameraData
from src.types.geometry import TriangulationOutput, RectifiedPointsOutput
from src.types.evaluation import (
    GeometryMetrics,
    ReprojectionMetrics,
    TrajectoryMetrics,
)
from src.geometry.triangulation import project_points, camera_frame_depths


def _index_points_by_frame(frames: Iterable) -> dict[int, list]:
    """Map `frame_index` → `points` for any sequence of rectified-output frames."""
    return {f.frame_index: f.points for f in frames}


def compute_reprojection_metrics(
    triangulations: TriangulationOutput,
    cam: CameraData,
    tracking_output: RectifiedPointsOutput,
) -> ReprojectionMetrics:
    """
    Project triangulated 3D points into cam's pixel space and compare against
    the 2D positions from the rectified tracking used to build the triangulations.
    Both inputs share the same frame_index domain so alignment is direct.

    The image-plane pixel error is converted to a physical distance by scaling each
    error with mm/px = Z / f, where Z is the point's camera-frame depth and
    f = (fx + fy) / 2 the focal length; this makes the metric resolution- and
    depth-independent and comparable across datasets and camera setups.
    Parameters:
        - triangulations: TriangulationOutput driving the frame clock.
        - cam: target camera for projection.
        - tracking_output: rectified tracking output used to build the triangulations.
    Returns:
        ReprojectionMetrics with error statistics (mm) and accuracy@25/50/100mm.
    """
    # Build indexes by actual frame_index so GT and triangulation stay aligned
    # even when the triangulation sequence contains every video frame.
    annotated_index = _index_points_by_frame(tracking_output.frames)
    # Average focal length (pixels); a 1 px offset spans Z / f_avg millimeters at depth Z.
    f_avg = float((cam.mtx[0, 0] + cam.mtx[1, 1]) / 2.0)
    # Compute reprojection error for each matched pair of 3D point and annotated 2D point.
    all_errors: list[float] = []

    for frame in triangulations.frames:
        if not frame.points:
            continue
        annotated_by_class = {d.class_name: d for d in annotated_index.get(frame.frame_index, [])}
        world_pts = np.array([[p.x, p.y, p.z] for p in frame.points], dtype=np.float32)
        pixels    = project_points(world_pts, cam)
        # Per-point pixel->mm scale at the point's camera-frame depth.
        mm_per_px = camera_frame_depths(world_pts, cam) / f_avg

        # For each projected point, if there is a corresponding annotated point with the same class_name
        for i, pt in enumerate(frame.points):
            pred = annotated_by_class.get(pt.class_name)
            if pred is not None:
                px, py = float(pixels[i, 0]), float(pixels[i, 1])
                error_px = float(np.sqrt((px - pred.x) ** 2 + (py - pred.y) ** 2))
                all_errors.append(error_px * float(mm_per_px[i]))

    # Compute error statistics (mm) and accuracy at different millimeter thresholds.
    errors = np.array(all_errors, dtype=np.float64)
    return ReprojectionMetrics(
        mean_error_mm     = float(np.mean(errors)),
        median_error_mm   = float(np.median(errors)),
        std_error_mm      = float(np.std(errors)),
        rmse_mm           = float(np.sqrt(np.mean(errors ** 2))),
        accuracy_at_25mm  = float(np.mean(errors < 25.0)),
        accuracy_at_50mm  = float(np.mean(errors < 50.0)),
        accuracy_at_100mm = float(np.mean(errors < 100.0)),
    )


def compute_trajectory_metrics(
    triangulations: TriangulationOutput,
    cam: CameraData,
    tracking_output: RectifiedPointsOutput,
    frame_stride: int = 5,
) -> TrajectoryMetrics:
    """
    Build per-identity GT and pred pixel trajectories, then compute ADE, FDE, MTE.
    Identities matched by class_name. Both inputs use absolute video frame indices,
    so alignment is by frame_index directly. frame_stride defines the gap threshold
    for fragment counting (consecutive GT frames are frame_stride apart).

    The image-plane displacement (pixels) is converted to a physical distance by
    scaling with mm/px = Z / f at the predicted point's camera-frame depth Z
    (f = (fx + fy) / 2), so the errors are reported in millimeters.
    Parameters:
        - triangulations: TriangulationOutput (predicted 3D positions).
        - cam: target camera for projection.
        - tracking_output: per-camera rectified annotated GT outputs.
        - frame_stride: GT annotation sampling rate (default 5).
    Returns:
        TrajectoryMetrics (mm) aggregated over all matched class_name trajectory pairs.
    """
    # Build the GT index by absolute frame_index so triangulation and GT stay
    # aligned even when the triangulation sequence covers every video frame.
    annotated_index = _index_points_by_frame(tracking_output.frames)
    # Average focal length (pixels); a 1 px offset spans Z / f_avg millimeters at depth Z.
    f_avg = float((cam.mtx[0, 0] + cam.mtx[1, 1]) / 2.0)

    # Build trajectory maps for triangulated and ground truth points.
    # The predicted entries also carry their pixel->mm scale (mm/px at the point's depth).
    triangulated_traj: dict[str, list[tuple[int, float, float, float]]] = defaultdict(list)
    gt_traj:           dict[str, list[tuple[int, float, float]]]        = defaultdict(list)

    # Iterate through triangulated frames and project points into pixel space, building trajectories keyed by class_name.
    for frame in triangulations.frames:
        if frame.points:
            world_pts = np.array([[p.x, p.y, p.z] for p in frame.points], dtype=np.float32)
            pixels    = project_points(world_pts, cam)
            mm_per_px = camera_frame_depths(world_pts, cam) / f_avg
            for i, pt in enumerate(frame.points):
                triangulated_traj[pt.class_name].append(
                    (frame.frame_index, float(pixels[i, 0]), float(pixels[i, 1]), float(mm_per_px[i]))
                )
        for d in annotated_index.get(frame.frame_index, []):
            gt_traj[d.class_name].append((frame.frame_index, float(d.x), float(d.y)))

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
        pred_pos_map = {fp: (x, y, s) for fp, x, y, s in pred_pts}

        # Find common frames between GT and predicted trajectories to compute errors.
        common_frames = sorted(gt_pos_map.keys() & pred_pos_map.keys())
        if not common_frames:
            continue

        # Compute metrics (mm) for the matched trajectory, scaling each pixel
        # displacement by the predicted point's mm/px factor at its depth.
        total_trajectories += 1
        errors = np.array([
            np.linalg.norm(np.array(gt_pos_map[f]) - np.array(pred_pos_map[f][:2])) * pred_pos_map[f][2]
            for f in common_frames
        ], dtype=np.float64)
        ade_values.append(float(np.mean(errors)))
        fde_values.append(float(errors[-1]))
        mte_values.append(float(np.median(errors)))

        if len(common_frames) >= 2:
            trajectory_fragments += sum(
                1 for a, b in zip(common_frames, common_frames[1:]) if b - a > frame_stride
            )

    return TrajectoryMetrics(
        ade_mm               = float(np.mean(ade_values)),
        fde_mm               = float(np.mean(fde_values)),
        mte_mm               = float(np.mean(mte_values)),
        total_trajectories   = total_trajectories,
        trajectory_fragments = trajectory_fragments,
    )


def evaluate_geometry(
    triangulations: TriangulationOutput,
    rectified_tracking: dict[str, RectifiedPointsOutput],
    rectified_annotated_tracking: dict[str, RectifiedPointsOutput],
    frame_stride: int = 5,
) -> dict[str, GeometryMetrics]:
    """
    Evaluate 3D tracking quality for each camera.

    For each camera the function computes:
      - ReprojectionMetrics: project triangulated 3D points back to image pixels
        and compare against rectified_tracking (the 2D input used to build triangulations).
      - TrajectoryMetrics: project triangulated 3D points to pixels and compare
        against rectified_annotated_tracking (GT, every frame_stride frames).

    Parameters:
        - triangulations: TriangulationOutput (predicted 3D positions).
        - rectified_tracking: camera_id → rectified tracking used to build the triangulations.
        - rectified_annotated_tracking: camera_id → rectified GT annotated tracking.
        - frame_stride: GT annotation sampling rate; used as gap threshold for fragment counting.
    Returns:
        Dict mapping camera_id → GeometryMetrics (reprojection + trajectory).
    """
    results: dict[str, GeometryMetrics] = {}
    # Iterate over cameras for which we have annotated GT tracking outputs, compute reprojection and trajectory metrics for each
    for camera_id, annotated_output in sorted(rectified_annotated_tracking.items()):
        cam = CameraData.load(camera_id)
        results[camera_id] = GeometryMetrics(
            reprojection = compute_reprojection_metrics(triangulations, cam, rectified_tracking[camera_id]),
            trajectory   = compute_trajectory_metrics(triangulations, cam, annotated_output, frame_stride),
        )

    return results
