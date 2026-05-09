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


@dataclass(frozen=True)
class _FrameMatch:
    """One matched GT↔pred pair within a single frame."""
    gt_class:   str
    pred_track: int
    error_px:   float   # Euclidean distance between projected GT and bbox centre


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
# Frame iterator  (mirrors _iter_frame_pairs in tracking_helpers)
# ---------------------------------------------------------------------------

def _iter_frame_pairs(
    triangulation: TriangulationOutput,
    projected_index: dict[int, list[_ProjectedPoint]],
    annotated_index: dict[int, list[TrackedDetection]],
    frame_stride: int,
) -> Iterator[tuple[int, list[_ProjectedPoint], list[TrackedDetection]]]:
    """Yield (frame_index, projected_points, annotated_detections) for each triangulation frame."""
    for gt_pos, frame in enumerate(triangulation.frames):
        projected = projected_index.get(frame.frame_index, [])
        annotated = annotated_index.get(gt_pos * frame_stride, [])
        yield frame.frame_index, projected, annotated


# ---------------------------------------------------------------------------
# Frame-level identity matching
# ---------------------------------------------------------------------------

def match_frame(
    projected_triangulation: list[_ProjectedPoint],
    annotated_detections: list[TrackedDetection],
) -> tuple[list[_FrameMatch], int, int]:
    """
    Match projected triangulation points to annotated detections by identity.
    Since class_name and track_id are already aligned by the algorithm, each
    GT point is looked up directly in the detection index — no spatial search
    needed.

    Parameters:
        - projected_triangulation: projected GT points for one frame.
        - annotated_detections: predicted detections for the same frame.
    Returns:
        matches        - list of _FrameMatch pairs (one per identity present in both)
        unmatched_gt   - number of triangulation points with no detection for that identity
        unmatched_pred - number of detections with no triangulation point for that identity
    """
    detection_by_id: dict[str, TrackedDetection] = {d.class_name: d for d in annotated_detections}

    matches:      list[_FrameMatch] = []
    matched_pred: set[int]          = set()

    for p in projected_triangulation:
        det = detection_by_id.get(p.class_name)
        if det is None:
            continue
        cx, cy = det.bbox.get_center()
        matches.append(_FrameMatch(
            gt_class   = p.class_name,
            pred_track = det.track_id,
            error_px   = float(np.linalg.norm([p.x - cx, p.y - cy])),
        ))
        matched_pred.add(det.track_id)

    unmatched_gt   = len(projected_triangulation) - len(matches)
    unmatched_pred = len(annotated_detections)    - len(matched_pred)
    return matches, unmatched_gt, unmatched_pred


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