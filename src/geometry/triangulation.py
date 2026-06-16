"""Multi-view triangulation of rectified detections into 3D world points.

Players are assumed to stand on the court plane (Z=0): each camera's foot pixel
is back-projected to a ray and intersected with the floor, then robustly combined
across cameras (median-inlier averaging). The ball is airborne, so it is
triangulated freely in 3D via DLT with reprojection-residual gating to reject
inconsistent views.
"""

from __future__ import annotations

import warnings

import cv2
import numpy as np

from src.calibration.camera_data import CameraData
from src.config import get_config
from src.types.geometry import (
    FrameTriangulatedPoints,
    Point3D,
    RectifiedPoint,
    RectifiedPointsOutput,
    TriangulationOutput,
)

def build_projection_matrix(
    mtx: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
) -> np.ndarray:
    """
    Build a 3x4 projection matrix P = K [R | t] from intrinsics and extrinsics.
    Parameters:
        - mtx (np.ndarray): 3x3 camera intrinsic matrix.
        - rvec (np.ndarray): Rotation vector (Rodrigues form), shape (3,) or (3, 1).
        - tvec (np.ndarray): Translation vector, shape (3,) or (3, 1).
    Returns:
        - P (np.ndarray): 3x4 projection matrix as float64.
    """
    R, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64))
    Rt = np.hstack([R, np.asarray(tvec, dtype=np.float64).reshape(3, 1)])
    return (np.asarray(mtx, dtype=np.float64) @ Rt)


def _camera_center_and_rotation(camera: CameraData) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(R, C)``: world->camera rotation and camera centre in world coords."""
    R, _ = cv2.Rodrigues(np.asarray(camera.rvec, dtype=np.float64))
    C = (-R.T @ np.asarray(camera.tvec, dtype=np.float64).reshape(3, 1)).reshape(3)
    return R, C


def _ground_point(camera: CameraData, pixel: np.ndarray) -> np.ndarray | None:
    """Back-project a rectified pixel to a ray and intersect it with Z=0.

    Returns a (3,) world point with z≈0, or None when the plane is behind the camera
    or the ray is parallel to the floor.
    """
    R, C = _camera_center_and_rotation(camera)
    # Ray direction in world frame: d = R^T K^{-1} [u, v, 1]
    K_inv = np.linalg.inv(np.asarray(camera.mtx, dtype=np.float64))
    ray_cam = K_inv @ np.array([pixel[0], pixel[1], 1.0])
    ray_world = R.T @ ray_cam
    if abs(ray_world[2]) < 1e-9:
        return None
    scale = -C[2] / ray_world[2]
    if scale <= 0:
        return None
    return C + scale * ray_world


def _combine_ground_points(estimates: list[np.ndarray], ground_inlier_mm: float) -> np.ndarray:
    """Robustly combine per-camera ground estimates: drop outliers, then mean."""
    if len(estimates) == 1:
        return estimates[0].copy()
    stacked = np.vstack(estimates)
    median = np.median(stacked, axis=0)
    distances = np.linalg.norm(stacked[:, :2] - median[:2], axis=1)
    inliers = stacked[distances <= ground_inlier_mm]
    return (inliers if len(inliers) > 0 else stacked).mean(axis=0)


def _dlt_triangulate(points_2d: list[np.ndarray], Ps: list[np.ndarray]) -> np.ndarray:
    """N-view DLT triangulation.  `points_2d` is a list of (2,) pixel coords."""
    rows = []
    for P, pt in zip(Ps, points_2d):
        x, y = pt
        rows.append(x * P[2] - P[0])
        rows.append(y * P[2] - P[1])
    A = np.stack(rows)
    _, _, Vt = np.linalg.svd(A)
    Xh = Vt[-1]
    return Xh[:3] / Xh[3]


def _reprojection_error(P: np.ndarray, X: np.ndarray, pt: np.ndarray) -> float:
    h = P @ np.append(X, 1.0)
    if abs(h[2]) < 1e-9:
        return float("inf")
    return float(np.linalg.norm(h[:2] / h[2] - pt))


def _triangulate_ball(
    Ps: list[np.ndarray],
    pixels: list[np.ndarray],
    ball_reproj_px: float,
    min_views: int,
) -> np.ndarray | None:
    """DLT-triangulate the ball, dropping views with large reprojection error."""
    if len(Ps) < min_views:
        return None
    X = _dlt_triangulate(pixels, Ps)
    errors = [_reprojection_error(P, X, px) for P, px in zip(Ps, pixels)]
    inliers = [i for i, e in enumerate(errors) if e <= ball_reproj_px]
    if len(inliers) < min_views:
        return None
    if len(inliers) < len(Ps):
        X = _dlt_triangulate([pixels[i] for i in inliers], [Ps[i] for i in inliers])
    return X


def triangulate_rectified_outputs(
    cameras: dict[str, CameraData],
    outputs: dict[str, RectifiedPointsOutput],
) -> TriangulationOutput:
    """
    Triangulate rectified 2D detections from multiple cameras into 3D world points.

    Players (non-ball classes) are placed on the court plane Z=0 via ground-ray
    intersection, with per-camera outlier rejection before averaging.  The ball is
    triangulated in full 3D via DLT with reprojection-residual gating.

    Cross-camera correspondence relies on class_id being unique per physical entity
    (ball + each individual player), so detections sharing (frame_index, class_id)
    across cameras refer to the same object.

    Parameters:
        - cameras (dict[str, CameraData]): Pre-loaded calibration per camera_id.
        - outputs (dict[str, RectifiedPointsOutput]): Map of camera_id -> rectified
          tracking output for the same synchronised game act. Keys must match `cameras`.
    Returns:
        - TriangulationOutput: Per-frame 3D points across all cameras.
    """
    if not outputs:
        warnings.warn(
            "triangulate_rectified_outputs called with no camera outputs; "
            "returning an empty TriangulationOutput.",
            stacklevel=2,
        )
        return TriangulationOutput(fps=0.0, camera_ids=[], frames=[])

    missing = set(outputs.keys()) - set(cameras.keys())
    if missing:
        raise KeyError(
            f"Calibration data missing for cameras present in outputs: {sorted(missing)}"
        )

    camera_ids = sorted(outputs.keys())
    Ps: dict[str, np.ndarray] = {
        cam_id: build_projection_matrix(cameras[cam_id].mtx, cameras[cam_id].rvec, cameras[cam_id].tvec)
        for cam_id in camera_ids
    }

    fps = outputs[camera_ids[0]].fps

    # Build per-camera index: frame_index -> {class_id: RectifiedPoint}.
    per_camera_index: dict[str, dict[int, dict[int, RectifiedPoint]]] = {}
    all_frame_indices: set[int] = set()
    for cam_id in camera_ids:
        frame_map: dict[int, dict[int, RectifiedPoint]] = {}
        for frame in outputs[cam_id].frames:
            class_map: dict[int, RectifiedPoint] = {}
            for pt in frame.points:
                class_map[pt.class_id] = pt
            frame_map[frame.frame_index] = class_map
            all_frame_indices.add(frame.frame_index)
        per_camera_index[cam_id] = frame_map

    tri_cfg = get_config().geometry.triangulation
    ball_class_id = get_config().classes.ball_class_id

    triangulated_frames: list[FrameTriangulatedPoints] = []
    for frame_index in sorted(all_frame_indices):
        # Gather observations: class_id -> list of (cam_id, RectifiedPoint).
        class_obs: dict[int, list[tuple[str, RectifiedPoint]]] = {}
        for cam_id in camera_ids:
            for class_id, pt in per_camera_index[cam_id].get(frame_index, {}).items():
                class_obs.setdefault(class_id, []).append((cam_id, pt))

        points_3d: list[Point3D] = []
        for class_id, observations in class_obs.items():
            pixels = [np.array([pt.x, pt.y], dtype=np.float64) for _, pt in observations]
            class_name = observations[0][1].class_name

            if class_id == ball_class_id:
                ball_Ps = [Ps[cam_id] for cam_id, _ in observations]
                X = _triangulate_ball(
                    ball_Ps, pixels,
                    ball_reproj_px=tri_cfg.ball_reproj_px,
                    min_views=tri_cfg.min_views,
                )
                if X is None:
                    continue
                points_3d.append(Point3D(
                    x=float(X[0]), y=float(X[1]), z=float(X[2]),
                    class_id=class_id, class_name=class_name,
                ))
            else:
                ground = [
                    g
                    for (cam_id, _), px in zip(observations, pixels)
                    if (g := _ground_point(cameras[cam_id], px)) is not None
                ]
                if not ground:
                    continue
                X = _combine_ground_points(ground, tri_cfg.ground_inlier_mm)
                points_3d.append(Point3D(
                    x=float(X[0]), y=float(X[1]), z=0.0,
                    class_id=class_id, class_name=class_name,
                ))

        triangulated_frames.append(FrameTriangulatedPoints(
            frame_index=frame_index, points=points_3d,
        ))

    if triangulated_frames and not any(f.points for f in triangulated_frames):
        warnings.warn(
            "triangulate_rectified_outputs produced no 3D points across "
            f"{len(triangulated_frames)} frames — no class id was observed in "
            "two or more cameras simultaneously. Check calibration and class id "
            "consistency across camera tracking outputs.",
            stacklevel=2,
        )

    return TriangulationOutput(fps=fps, camera_ids=camera_ids, frames=triangulated_frames)


def project_points(world_points: np.ndarray, cam: CameraData) -> np.ndarray:
    """
    Project (N, 3) world-frame points into the pixel space of `cam`.
    Distortion coefficients are zeroed out — inputs are pre-rectified.
    """
    image_points, _ = cv2.projectPoints(
        world_points,
        cam.rvec,
        cam.tvec,
        cam.mtx,
        np.zeros_like(cam.dist),
    )
    return image_points.reshape(-1, 2)


def camera_frame_depths(world_points: np.ndarray, cam: CameraData) -> np.ndarray:
    """
    Return the camera-frame depth Z (millimeters) of each (N, 3) world point.

    Used to convert an image-plane pixel error into a physical distance at the
    object's depth: a 1 px offset spans ≈ Z / f millimeters on the fronto-parallel
    plane at depth Z, where f is the focal length in pixels.
    """
    R, _ = cv2.Rodrigues(np.asarray(cam.rvec, dtype=np.float64))
    cam_pts = R @ np.asarray(world_points, dtype=np.float64).T + np.asarray(cam.tvec, dtype=np.float64).reshape(3, 1)
    return cam_pts[2, :]
