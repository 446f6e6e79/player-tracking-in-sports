import cv2
import numpy as np

from src.calibration.camera_data import CameraData
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
        - P (np.ndarray): 3x4 projection matrix as float32.
    """
    R, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float32))
    Rt = np.hstack([R, np.asarray(tvec, dtype=np.float32).reshape(3, 1)])
    return (mtx.astype(np.float32) @ Rt).astype(np.float32)


def _triangulate_point(
    points_2d: np.ndarray,
    projection_matrices: np.ndarray,
) -> np.ndarray:
    """
    N-view triangulation via DLT. Solves the linear system formed by stacking
    x_i × (P_i X) = 0 for each view, then takes the smallest singular vector.
    For M==2 this matches cv2.triangulatePoints; for M>2 it generalises naturally.
    Parameters:
        - points_2d (np.ndarray): (M, 2) pixel coordinates of the same physical
          point in M cameras (rectified pixels in the original K frame).
        - projection_matrices (np.ndarray): (M, 3, 4) projection matrices.
    Returns:
        - X (np.ndarray): (3,) world-frame coordinates as float32.
    """
    pts = np.asarray(points_2d, dtype=np.float64).reshape(-1, 2)
    Ps = np.asarray(projection_matrices, dtype=np.float64).reshape(-1, 3, 4)
    M = pts.shape[0]
    A = np.zeros((2 * M, 4), dtype=np.float64)
    for i in range(M):
        x, y = pts[i]
        P = Ps[i]
        A[2 * i] = x * P[2] - P[0]
        A[2 * i + 1] = y * P[2] - P[1]
    _, _, Vt = np.linalg.svd(A)
    X_h = Vt[-1]
    return (X_h[:3] / X_h[3]).astype(np.float32)


def triangulate_rectified_outputs(
    cameras: dict[str, CameraData],
    outputs: dict[str, RectifiedPointsOutput],
) -> TriangulationOutput:
    """
    Triangulate rectified 2D detections from multiple cameras into 3D world points.
    Cross-camera correspondence relies on class_id being unique per physical entity
    (ball + each individual player), so detections sharing (frame_index, class_id)
    across cameras refer to the same object. Class ids visible in only one camera
    are skipped.
    Parameters:
        - cameras (dict[str, CameraData]): Pre-loaded calibration per camera_id.
        - outputs (dict[str, RectifiedPointsOutput]): Map of camera_id -> rectified
          tracking output for the same synchronised game act. Keys must match `cameras`.
    Returns:
        - TriangulationOutput: Per-frame 3D points across all cameras.
    """
    if not outputs:
        return TriangulationOutput(fps=0.0, camera_ids=[], frames=[])

    # Stable camera order, plus per-camera projection matrices and intrinsics.
    camera_ids = sorted(outputs.keys())
    projections: dict[str, np.ndarray] = {
        cam_id: build_projection_matrix(cameras[cam_id].mtx, cameras[cam_id].rvec, cameras[cam_id].tvec)
        for cam_id in camera_ids
    }

    # Sanity: all outputs should share fps; take the first as canonical.
    fps = outputs[camera_ids[0]].fps

    # Build per-camera index: frame_index -> {class_id: RectifiedPoint}.
    per_camera_index: dict[str, dict[int, dict[int, object]]] = {}
    all_frame_indices: set[int] = set()
    for cam_id in camera_ids:
        frame_index_to_class_map: dict[int, dict[int, object]] = {}
        for frame in outputs[cam_id].frames:
            class_to_point: dict[int, object] = {}
            for pt in frame.points:
                class_to_point[pt.class_id] = pt
            frame_index_to_class_map[frame.frame_index] = class_to_point
            all_frame_indices.add(frame.frame_index)
        per_camera_index[cam_id] = frame_index_to_class_map

    triangulated_frames: list[FrameTriangulatedPoints] = []
    for frame_index in sorted(all_frame_indices):
        # Gather class_ids visible in this frame across cameras.
        class_to_observations: dict[int, list[tuple[str, object]]] = {}
        for cam_id in camera_ids:
            class_to_point = per_camera_index[cam_id].get(frame_index, {})
            for class_id, pt in class_to_point.items():
                class_to_observations.setdefault(class_id, []).append((cam_id, pt))

        points_3d: list[Point3D] = []
        for class_id, observations in class_to_observations.items():
            if len(observations) < 2:
                continue
            pts_2d = np.array([[pt.x, pt.y] for _, pt in observations], dtype=np.float64)
            Ps = np.stack([projections[cam_id] for cam_id, _ in observations], axis=0)
            X = _triangulate_point(pts_2d, Ps)
            class_name = observations[0][1].class_name
            points_3d.append(Point3D(
                x=float(X[0]),
                y=float(X[1]),
                z=float(X[2]),
                class_id=class_id,
                class_name=class_name,
            ))

        triangulated_frames.append(FrameTriangulatedPoints(
            frame_index=frame_index,
            points=points_3d,
        ))

    return TriangulationOutput(
        fps=fps,
        camera_ids=camera_ids,
        frames=triangulated_frames,
    )


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


def project_triangulation(
    triangulation: TriangulationOutput,
    cam: CameraData,
) -> dict[int, list[RectifiedPoint]]:
    """
    Project all 3D points from `triangulation` into `cam`'s pixel space.
    Returns frame_index → list[RectifiedPoint], preserving point order within
    each frame (matches `triangulation.frames[*].points`).
    """
    index: dict[int, list[RectifiedPoint]] = {}
    # Build an index of annotated points by frame for easy lookup.
    for frame in triangulation.frames:
        # Project all 3D points in the frame into pixel space using the camera parameters.
        world_pts = np.array([[p.x, p.y, p.z] for p in frame.points], dtype=np.float32)
        pixels    = project_points(world_pts, cam)
        # For each projected point, create a corresponding RectifiedPoint with the same class_id and class_name
        index[frame.frame_index] = [
            RectifiedPoint(
                x          = float(pixels[i, 0]),
                y          = float(pixels[i, 1]),
                confidence = 1.0,
                class_id   = getattr(pt, "class_id", -1),
                class_name = pt.class_name,
                track_id   = -1,
            )
            for i, pt in enumerate(frame.points)
        ]
    return index
