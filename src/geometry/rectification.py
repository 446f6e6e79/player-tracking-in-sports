import json
import os

import cv2
import numpy as np

from src.types.geometry import FrameRectifiedPoints, RectifiedPoint, RectifiedPointsOutput
from src.types.tracking import TrackingOutput


def load_calibration(calib_path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Load the camera intrinsics and distortion coefficients from a JSON file.
    Parameters:
        - calib_path (str): Path to a calibration JSON file containing "mtx" and "dist" entries.
    Returns:
        - mtx (np.ndarray): 3x3 camera matrix as float32.
        - dist (np.ndarray): Distortion coefficients as float32.
    """
    if not os.path.exists(calib_path):
        raise FileNotFoundError(f"Calibration file not found: {calib_path}")

    with open(calib_path, "r") as f:
        calib = json.load(f)

    mtx = np.array(calib["mtx"], dtype=np.float32)
    dist = np.array(calib["dist"], dtype=np.float32)
    return mtx, dist


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
    # OpenCV expects points in shape (N, 1, 2) for undistortPoints, and outputs the same shape.
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
    
    rectified = cv2.undistortPoints(pts, mtx, dist, P=mtx)
    return rectified.reshape(-1, 2)


def rectify_tracking_output(tracking: TrackingOutput, calib_path: str) -> RectifiedPointsOutput:
    """
    Rectify the bounding box centers from a TrackingOutput using camera calibration.
    Parameters:
        - tracking (TrackingOutput): The input tracking output containing frames and detections.
        - calib_path (str): Path to the calibration JSON file.
    Returns:
        - RectifiedPointsOutput: A new output containing rectified points for each detection.
    """
    # Load camera calibration parameters
    mtx, dist = load_calibration(calib_path)

    # Collect all bounding box centers across all frames
    centers: list[tuple[float, float]] = []
    for frame in tracking.frames:
        for det in frame.detections:
            centers.append(det.bbox.get_center())

    # Rectify all centers in a single batch for efficiency
    centers_array = np.array(centers, dtype=np.float32) if centers else np.empty((0, 2), dtype=np.float32)
    rectified_array = rectify_points(centers_array, mtx, dist)

    # Reconstruct the output frames with rectified points, preserving the original structure
    frames: list[FrameRectifiedPoints] = []
    base_frame_index = 0

    for frame in tracking.frames:
        points: list[RectifiedPoint] = []

        for offset, det in enumerate(frame.detections):
            x_rect, y_rect = rectified_array[base_frame_index + offset]
            points.append(RectifiedPoint(
                x=float(x_rect),
                y=float(y_rect),
                confidence=det.confidence,
                class_id=det.class_id,
                class_name=det.class_name,
                track_id=det.track_id,
            ))
        base_frame_index += len(frame.detections)
        frames.append(FrameRectifiedPoints(frame_index=frame.frame_index, points=points))

    return RectifiedPointsOutput(
        source=tracking.source,
        camera_id=tracking.camera_id,
        fps=tracking.fps,
        frames=frames,
    )
