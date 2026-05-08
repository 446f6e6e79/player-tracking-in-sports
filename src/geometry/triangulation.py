import cv2
import numpy as np

from src.geometry.load_camera_data import (
    get_distortion,
    get_extrinsics,
    get_intrinsics,
    load_camera_data,
)
from src.types.geometry import (
    FrameTriangulatedPoints,
    Point3D,
    RectifiedPointsOutput,
    TriangulationOutput,
)


def compute_extrinsics(
    world_points: np.ndarray,
    image_points: np.ndarray,
    mtx: np.ndarray,
    dist: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute camera extrinsics from N>=4 court reference correspondences via solvePnP.
    Parameters:
        - world_points (np.ndarray): (N, 3) reference points in world (court) coordinates.
        - image_points (np.ndarray): (N, 2) matching pixel coordinates in the raw,
          distorted image (solvePnP applies the distortion model internally).
        - mtx (np.ndarray): 3x3 camera intrinsic matrix.
        - dist (np.ndarray): Distortion coefficients.
    Returns:
        - rvec (np.ndarray): (3, 1) rotation vector (Rodrigues form), float32.
        - tvec (np.ndarray): (3, 1) translation vector, float32.
    """
    obj_pts = np.asarray(world_points, dtype=np.float32).reshape(-1, 3)
    img_pts = np.asarray(image_points, dtype=np.float32).reshape(-1, 2)
    ok, rvec, tvec = cv2.solvePnP(obj_pts, img_pts, mtx, dist, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        raise RuntimeError("solvePnP failed to converge")
    return rvec.astype(np.float32), tvec.astype(np.float32)


def recover_relative_pose(
    pts_a: np.ndarray,
    pts_b: np.ndarray,
    mtx_a: np.ndarray,
    dist_a: np.ndarray,
    mtx_b: np.ndarray,
    dist_b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Recover camera B's pose relative to A from cross-view correspondences.

    Both point sets are first undistorted to normalized image rays, so cameras
    with different intrinsics and distortion are handled correctly. The returned
    pose has a unit translation (‖t‖ = 1) — multi-view calibration callers must
    set the absolute scale separately.

    Parameters:
        - pts_a (np.ndarray): (N, 2) raw distorted pixel coordinates in camera A.
        - pts_b (np.ndarray): (N, 2) same physical points in camera B.
        - mtx_a, dist_a: intrinsics and distortion of camera A.
        - mtx_b, dist_b: intrinsics and distortion of camera B.
    Returns:
        - R (np.ndarray): (3, 3) rotation, camera B in camera A's frame.
        - t (np.ndarray): (3, 1) translation, ‖t‖ = 1.
        - inlier_mask (np.ndarray): (N,) uint8, 1 for inliers used by recoverPose.
        - pts_a_norm (np.ndarray): (N, 2) normalized rays for camera A.
        - pts_b_norm (np.ndarray): (N, 2) normalized rays for camera B.
    """
    pa = np.asarray(pts_a, dtype=np.float32).reshape(-1, 1, 2)
    pb = np.asarray(pts_b, dtype=np.float32).reshape(-1, 1, 2)
    pa_norm = cv2.undistortPoints(pa, mtx_a, dist_a).reshape(-1, 2)
    pb_norm = cv2.undistortPoints(pb, mtx_b, dist_b).reshape(-1, 2)

    K_id = np.eye(3, dtype=np.float64)
    E, mask = cv2.findEssentialMat(
        pa_norm, pb_norm, K_id,
        method=cv2.RANSAC, prob=0.999, threshold=1e-3,
    )
    if E is None:
        raise RuntimeError("findEssentialMat failed — too few inliers or degenerate configuration")
    _, R, t, pose_mask = cv2.recoverPose(E, pa_norm, pb_norm, K_id, mask=mask)
    return (
        R.astype(np.float32),
        t.astype(np.float32).reshape(3, 1),
        pose_mask.reshape(-1),
        pa_norm.astype(np.float32),
        pb_norm.astype(np.float32),
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


def triangulate_point(
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
    outputs: dict[str, RectifiedPointsOutput],
) -> TriangulationOutput:
    """
    Triangulate rectified 2D detections from multiple cameras into 3D world points.
    Cross-camera correspondence relies on class_id being unique per physical entity
    (ball + each individual player), so detections sharing (frame_index, class_id)
    across cameras refer to the same object. Class ids visible in only one camera
    are skipped.
    Parameters:
        - outputs (dict[str, RectifiedPointsOutput]): Map of camera_id -> rectified
          tracking output for the same synchronised game act.
    Returns:
        - TriangulationOutput: Per-frame 3D points across all cameras.
    """
    if not outputs:
        return TriangulationOutput(fps=0.0, camera_ids=[], frames=[])

    # Stable camera order, plus per-camera projection matrices and intrinsics.
    camera_ids = sorted(outputs.keys())
    projections: dict[str, np.ndarray] = {}
    for cam_id in camera_ids:
        data = load_camera_data(cam_id)
        mtx = get_intrinsics(data)
        rvec, tvec = get_extrinsics(data)
        projections[cam_id] = build_projection_matrix(mtx, rvec, tvec)

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
            X = triangulate_point(pts_2d, Ps)
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
