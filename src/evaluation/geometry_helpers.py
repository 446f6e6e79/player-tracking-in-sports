from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass

import cv2
import numpy as np

from src.calibration.camera_data import CameraData
from src.types.geometry import TriangulationOutput
from src.types.tracking import TrackingOutput, TrackedDetection


# ---------------------------------------------------------------------------
# Projected point and frame match
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _ProjectedPoint:
    """Projected GT pixel position for one identity in one frame."""
    x: float
    y: float
    class_name: str   # GT identity key, e.g. "White_14"

# ---------------------------------------------------------------------------
# Projection
# ---------------------------------------------------------------------------

def _project_points(world_points: np.ndarray, cam: CameraData) -> np.ndarray:
    """
    Project (N, 3) world-frame points into the distorted pixel space of `cam`.
    Parameters:
        - world_points: (N, 3) float32 in world coordinates (mm).
        - cam: CameraData with intrinsics and extrinsics.
    Returns:
        (N, 2) float32 pixel coordinates in the original (distorted) image frame.
    """
    image_points, _ = cv2.projectPoints(
        world_points,
        cam.rvec,
        cam.tvec,
        cam.mtx,
        cam.dist
    )
    return image_points.reshape(-1, 2)


def project_triangulation(
    triangulation: TriangulationOutput,
    cam: CameraData,
) -> dict[int, list[_ProjectedPoint]]:
    """
    Project all 3D points from `triangulation` into `cam`'s pixel space.
    Parameters:
        - triangulation: annotated TriangulationOutput (GT identity and 3D positions).
        - cam: CameraData for the target camera view.
    Returns:
        frame_index → list[_ProjectedPoint], preserving point order within each
        frame (matches `triangulation.frames[*].points`).
    """
    index: dict[int, list[_ProjectedPoint]] = {}

    for frame in triangulation.frames:
        if not frame.points:
            index[frame.frame_index] = []
            continue

        world_pts = np.array([[p.x, p.y, p.z] for p in frame.points], dtype=np.float32)
        pixels    = _project_points(world_pts, cam)

        index[frame.frame_index] = [
            _ProjectedPoint(
                x          = float(pixels[i, 0]),
                y          = float(pixels[i, 1]),
                class_name = pt.class_name,
            )
            for i, pt in enumerate(frame.points)
        ]

    return index

# ---------------------------------------------------------------------------
# Index builder  (mirrors pred_index pattern in tracking_helpers)
# ---------------------------------------------------------------------------

def build_annotated_index(
    annotated_tracking: TrackingOutput,
) -> dict[int, list[TrackedDetection]]:
    """
    Build a frame_index → detections lookup from an annotated TrackingOutput.
    Parameters:
        - annotated_tracking: per-camera TrackingOutput to index.
    Returns:
        frame_index → list[TrackedDetection].
    """
    return {frame.frame_index: frame.detections for frame in annotated_tracking.frames}