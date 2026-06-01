import json
from pathlib import Path
import cv2
import numpy as np

from src.geometry.court import LANDMARKS, WORLD_LANDMARKS_MM

# Canonical home for the picker output type — picker.py imports it from here
# so that extrinsics doesn't depend on the UI module.
LandmarkClicks = dict[str, tuple[float, float] | None]


def _compute_extrinsics(
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
    real_points = np.asarray(world_points, dtype=np.float32).reshape(-1, 3)
    img_points = np.asarray(image_points, dtype=np.float32).reshape(-1, 2)
    # Solve the PnP problem between the clicked landmarks and their known world coordinates, using the provided intrinsics.
    ok, rvec, tvec = cv2.solvePnP(real_points, img_points, mtx, dist, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        raise RuntimeError("solvePnP failed to converge")
    return rvec.astype(np.float32), tvec.astype(np.float32)


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
        - residuals_px: (N,) reprojection error per landmark (pixels), same order as labels.
    Raises:
        ValueError if fewer than `min_points` landmarks have both a click and a
        world coordinate.
    """
    # Filter out unclicked landmarks
    labels = [
        landmark for landmark in LANDMARKS
        if clicks.get(landmark) is not None
    ]
    # Check that we have enough points to solvePnP
    if len(labels) < min_points:
        raise ValueError(
            f"Need >={min_points} clicked landmarks with known world coords, got {len(labels)}."
        )

    # For each clicked landmark, gather its world coordinate and the clicked pixel coordinate.
    world_pts = np.stack([WORLD_LANDMARKS_MM[landmark] for landmark in labels], axis=0)
    img_pts = np.array([clicks[landmark] for landmark in labels], dtype=np.float32)

    # Compute the extrinsics via solvePnP
    rvec, tvec = _compute_extrinsics(world_pts, img_pts, mtx, dist)

    # Reproject the world points back into the image and compute residuals
    proj, _ = cv2.projectPoints(world_pts, rvec, tvec, mtx, dist)
    residuals_px = np.linalg.norm(proj.reshape(-1, 2) - img_pts, axis=1).astype(np.float32)
    return rvec, tvec, labels, residuals_px


def reprojection_rmse(
    rvec: np.ndarray,
    tvec: np.ndarray,
    mtx: np.ndarray,
    dist: np.ndarray,
    world_points: np.ndarray,
    image_points: np.ndarray,
) -> float:
    """Return the reprojection RMSE (pixels) for a set of point correspondences."""
    projected, _ = cv2.projectPoints(
        np.asarray(world_points, dtype=np.float32).reshape(-1, 1, 3),
        rvec, tvec, mtx, dist,
    )
    errors = np.linalg.norm(projected.reshape(-1, 2) - np.asarray(image_points, dtype=np.float32).reshape(-1, 2), axis=1)
    return float(np.sqrt(np.mean(errors**2)))


def verify_stored_extrinsics(camera_data_path: Path) -> float:
    """Compute reprojection RMSE (px) for stored extrinsics + saved landmark pixels.

    Raises KeyError if the camera JSON has no ``landmarks_px`` key (saved only
    after re-running ``calibrate_extrinsics.py`` with the updated script).
    """
    with open(camera_data_path) as f:
        raw = json.load(f)
    if "landmarks_px" not in raw:
        raise KeyError(
            f"{camera_data_path} has no 'landmarks_px'; re-run calibration to enable verification"
        )
    mtx = np.array(raw["mtx"], dtype=np.float32)
    dist = np.array(raw["dist"], dtype=np.float32)
    rvec = np.array(raw["rvecs"], dtype=np.float32).reshape(3, 1)
    tvec = np.array(raw["tvecs"], dtype=np.float32).reshape(3, 1)
    world_pts = []
    img_pts = []
    for name, px in raw["landmarks_px"].items():
        if name in WORLD_LANDMARKS_MM:
            world_pts.append(WORLD_LANDMARKS_MM[name])
            img_pts.append(px)
    if len(world_pts) < 4:
        raise ValueError("Not enough stored landmarks to compute RMSE (need ≥ 4).")
    return reprojection_rmse(rvec, tvec, mtx, dist, np.array(world_pts), np.array(img_pts))
