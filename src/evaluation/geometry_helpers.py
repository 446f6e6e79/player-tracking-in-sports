import cv2
import numpy as np

from src.calibration.camera_data import CameraData
from src.types.geometry import TriangulationOutput, RectifiedPoint


# ---------------------------------------------------------------------------
# Projection
# ---------------------------------------------------------------------------

def _project_points(world_points: np.ndarray, cam: CameraData) -> np.ndarray:
    """
    Project (N, 3) world-frame points into the pixel space of `cam`.
    Parameters:
        - world_points: (N, 3) float32 in world coordinates (mm).
        - cam: CameraData with intrinsics and extrinsics.
    Returns:
        (N, 2) float32 pixel coordinates in the requested pixel frame.
    """
    image_points, _ = cv2.projectPoints(
        world_points,
        cam.rvec,
        cam.tvec,
        cam.mtx,
        np.zeros_like(cam.dist)
    )
    return image_points.reshape(-1, 2)


def project_triangulation(
    triangulation: TriangulationOutput,
    cam: CameraData,
) -> dict[int, list[RectifiedPoint]]:
    """
    Project all 3D points from `triangulation` into `cam`'s pixel space.
    Parameters:
        - triangulation: annotated TriangulationOutput (GT identity and 3D positions).
        - cam: CameraData for the target camera view.
    Returns:
        frame_index → list[_ProjectedPoint], preserving point order within each
        frame (matches `triangulation.frames[*].points`).
    """
    index: dict[int, list[RectifiedPoint]] = {}

    for frame in triangulation.frames:
        if not frame.points:
            index[frame.frame_index] = []
            continue

        world_pts = np.array([[p.x, p.y, p.z] for p in frame.points], dtype=np.float32)
        pixels    = _project_points(world_pts, cam)

        index[frame.frame_index] = [
            RectifiedPoint(
                x= float(pixels[i, 0]),
                y= float(pixels[i, 1]),
                confidence= 1.0,
                class_id= getattr(pt, 'class_id', -1),
                class_name= pt.class_name,
                track_id= -1,
            )
            for i, pt in enumerate(frame.points)
        ]

    return index