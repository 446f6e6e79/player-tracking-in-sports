import cv2
import numpy as np

from src.geometry.calibration_landmarks import (
    HOOP_HEIGHT_MM,
    LANDMARKS,
    LandmarkClicks,
    SCALE_PAIRS,
)
from src.geometry.triangulation import recover_relative_pose


def common_labels(clicks_a: LandmarkClicks, clicks_b: LandmarkClicks) -> list[str]:
    """Landmarks clicked in both views, in canonical LANDMARKS order."""
    return [lbl for lbl in LANDMARKS if clicks_a.get(lbl) is not None and clicks_b.get(lbl) is not None]


def stack_clicks(clicks: LandmarkClicks, labels: list[str]) -> np.ndarray:
    """(N, 2) float32 array of click coordinates for `labels`."""
    return np.array([clicks[l] for l in labels], dtype=np.float32)


def triangulate_landmark_pair(
    clicks_anchor: LandmarkClicks,
    clicks_b: LandmarkClicks,
    mtx_anchor: np.ndarray,
    dist_anchor: np.ndarray,
    mtx_b: np.ndarray,
    dist_b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], int, int]:
    """
    Recover camera B's pose relative to the anchor and triangulate the matched
    landmarks in the anchor's frame at arbitrary scale (‖t‖ = 1).

    Returns:
        R_b: (3, 3) rotation, camera B in anchor frame.
        t_b: (3, 1) translation, ‖t‖ = 1.
        structure: {label: (3,) np.ndarray} for inlier landmarks only.
        n_inliers: number of recoverPose inliers.
        n_pairs: number of common labels considered.
    """
    pair_labels = common_labels(clicks_anchor, clicks_b)
    pts_anchor = stack_clicks(clicks_anchor, pair_labels)
    pts_b = stack_clicks(clicks_b, pair_labels)

    R_b, t_b, inlier_mask, p_anchor_norm, p_b_norm = recover_relative_pose(
        pts_anchor, pts_b, mtx_anchor, dist_anchor, mtx_b, dist_b,
    )

    P_anchor = np.hstack([np.eye(3), np.zeros((3, 1))]).astype(np.float32)
    P_b = np.hstack([R_b, t_b]).astype(np.float32)
    Xh = cv2.triangulatePoints(P_anchor, P_b, p_anchor_norm.T, p_b_norm.T)
    X = (Xh[:3] / Xh[3]).T

    structure = {
        label: X[i] for i, label in enumerate(pair_labels) if inlier_mask[i]
    }
    return R_b, t_b, structure, int(inlier_mask.sum()), len(pair_labels)


def recover_metric_scale(
    structure: dict[str, np.ndarray],
    scale_pairs: list[tuple[str, str]] = SCALE_PAIRS,
    target_mm: float = HOOP_HEIGHT_MM,
) -> tuple[float, float] | None:
    """
    Solve for the scalar that turns the arbitrary-scale anchor structure into
    millimeters, using the mean of the available (top, bottom) physical pairs.

    Returns (scale, mean_distance_arbitrary_units) or None when no pair was
    triangulated.
    """
    distances = [
        float(np.linalg.norm(structure[top] - structure[bottom]))
        for top, bottom in scale_pairs
        if top in structure and bottom in structure
    ]
    if not distances:
        return None
    mean_d = float(np.mean(distances))
    return target_mm / mean_d, mean_d


def build_extrinsics(
    anchor_id: str,
    cam_b_id: str,
    R_b: np.ndarray,
    t_b_scaled: np.ndarray,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Pack the (rvec, tvec) pairs for both cameras in the anchor's world frame."""
    rvec_b, _ = cv2.Rodrigues(R_b)
    return {
        anchor_id: (
            np.zeros((3, 1), dtype=np.float32),
            np.zeros((3, 1), dtype=np.float32),
        ),
        cam_b_id: (
            rvec_b.astype(np.float32),
            t_b_scaled.astype(np.float32),
        ),
    }


def reprojection_residuals(
    structure_scaled: dict[str, np.ndarray],
    clicks: LandmarkClicks,
    rvec: np.ndarray,
    tvec: np.ndarray,
    mtx: np.ndarray,
    dist: np.ndarray,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """
    Project the scaled 3D structure into one camera and measure the per-landmark
    pixel error against the user's clicks.

    Returns (labels, residuals_px (N,), projected_pts (N, 2)). `labels` are the
    landmarks present both in the structure and in this camera's clicks.
    """
    labels = [l for l in LANDMARKS if l in structure_scaled and clicks.get(l) is not None]
    if not labels:
        return [], np.empty((0,), dtype=np.float32), np.empty((0, 2), dtype=np.float32)
    world = np.array([structure_scaled[l] for l in labels], dtype=np.float32)
    clicked = stack_clicks(clicks, labels)
    proj, _ = cv2.projectPoints(world, rvec, tvec, mtx, dist)
    proj = proj.reshape(-1, 2)
    residuals = np.linalg.norm(proj - clicked, axis=1)
    return labels, residuals, proj
