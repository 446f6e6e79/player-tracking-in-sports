import cv2
import numpy as np

from src.calibration.camera_data import CameraData
from src.config import get_config
from src.types.geometry import FrameRectifiedPoints, RectifiedPoint, RectifiedPointsOutput
from src.types.tracking import TrackingOutput


def rectify_points(points: np.ndarray, mtx: np.ndarray, dist: np.ndarray) -> np.ndarray:
    """
    Undistort a batch of pixel points using the camera intrinsics and distortion
    coefficients. Output points are expressed in the same rectified pixel frame
    (P=mtx), so they remain directly comparable to the original image space.
    Parameters:
        - points (np.ndarray): Input pixel points of shape (N, 2).
        - mtx (np.ndarray): 3x3 camera matrix.
        - dist (np.ndarray): Distortion coefficients.
    Returns:
        - rectified (np.ndarray): Output points of shape (N, 2), float32.
    """
    if points.size == 0:
        return np.empty((0, 2), dtype=np.float32)
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
    rectified = cv2.undistortPoints(pts, mtx, dist, P=mtx)
    return rectified.reshape(-1, 2)


def rectify_tracking_output(tracking: TrackingOutput, camera: CameraData) -> RectifiedPointsOutput:
    """
    Rectify the detection reference pixels from a TrackingOutput using camera calibration.

    The ball (class_id == 0) uses its bbox centre; all other detections (players)
    use their bbox bottom-centre — the ground-contact point — so they can be
    projected onto the court floor plane downstream.

    Parameters:
        - tracking (TrackingOutput): The input tracking output containing frames and detections.
        - camera (CameraData): Pre-loaded calibration for the camera that produced `tracking`.
    Returns:
        - RectifiedPointsOutput: A new output containing rectified points for each detection.
    """
    ball_class_id = get_config().classes.ball_class_id
    # Choose the reference pixel per detection (ball → centre; players → bottom-centre).
    ref_pixels: list[tuple[float, float]] = []
    for frame in tracking.frames:
        for det in frame.detections:
            if det.class_id == ball_class_id:
                ref_pixels.append(det.bbox.get_center())
            else:
                ref_pixels.append(det.bbox.get_bottom_center())

    # Rectify all reference pixels in a single batch for efficiency.
    pts_array = np.array(ref_pixels, dtype=np.float32) if ref_pixels else np.empty((0, 2), dtype=np.float32)
    rectified_array = rectify_points(pts_array, camera.mtx, camera.dist)

    # Reconstruct the output frames with rectified points, preserving the original structure.
    frames: list[FrameRectifiedPoints] = []
    offset = 0
    for frame in tracking.frames:
        points: list[RectifiedPoint] = []
        for det in frame.detections:
            x_rect, y_rect = rectified_array[offset]
            points.append(RectifiedPoint(
                x=float(x_rect),
                y=float(y_rect),
                confidence=det.confidence,
                class_id=det.class_id,
                class_name=det.class_name,
                track_id=det.track_id,
            ))
            offset += 1
        frames.append(FrameRectifiedPoints(frame_index=frame.frame_index, points=points))

    return RectifiedPointsOutput(
        source=tracking.source,
        camera_id=tracking.camera_id,
        fps=tracking.fps,
        frames=frames,
    )
