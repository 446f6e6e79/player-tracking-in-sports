import cv2
import numpy as np

from src.geometry.calibration_landmarks import LANDMARKS, LandmarkClicks
from src.geometry.court_model import WORLD_LANDMARKS_MM
from src.geometry.triangulation import compute_extrinsics


def solve_camera_pose(
    clicks: LandmarkClicks,
    mtx: np.ndarray,
    dist: np.ndarray,
    min_points: int = 6,
) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
    """
    Calibrate a single camera against the known 3D court model via solvePnP.

    Parameters:
        - clicks: landmark label -> (x, y) pixel | None for the camera frame.
        - mtx, dist: intrinsics of that camera.
        - min_points: minimum number of clicked landmarks required.
    Returns:
        - rvec: (3, 1) Rodrigues rotation, world (court) frame -> camera frame.
        - tvec: (3, 1) translation in millimeters.
        - labels: landmarks used in the solve, in click order.
        - residuals_px: (N,) reprojection error per landmark, same order as labels.
    Raises:
        ValueError if fewer than `min_points` landmarks have both a click and a
        world coordinate.
    """
    labels = [
        l for l in LANDMARKS
        if clicks.get(l) is not None and l in WORLD_LANDMARKS_MM
    ]
    if len(labels) < min_points:
        raise ValueError(
            f"Need >={min_points} clicked landmarks with known world coords, got {len(labels)}."
        )

    world_pts = np.stack([WORLD_LANDMARKS_MM[l] for l in labels], axis=0)
    img_pts = np.array([clicks[l] for l in labels], dtype=np.float32)

    rvec, tvec = compute_extrinsics(world_pts, img_pts, mtx, dist)

    proj, _ = cv2.projectPoints(world_pts, rvec, tvec, mtx, dist)
    residuals_px = np.linalg.norm(proj.reshape(-1, 2) - img_pts, axis=1).astype(np.float32)
    return rvec, tvec, labels, residuals_px
