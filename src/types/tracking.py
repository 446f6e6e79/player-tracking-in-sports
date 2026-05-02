from dataclasses import dataclass, field
import numpy as np


@dataclass
class BoundingBox:
    """Represents a bounding box in pixel coordinates."""
    x1: float
    y1: float
    x2: float
    y2: float

@dataclass
class Detection:
    """Represents a single object detection in a frame, optionally with a track ID."""
    bbox: BoundingBox             # Bounding box coordinates (x1, y1, x2, y2)
    confidence: float             # Confidence score of the detection (0.0 to 1.0)
    class_id: int                 # Class ID of the detection
    class_name: str               # Human-readable class name (e.g. "player")
    track_id: int | None = None   # Tracker-assigned ID; None when no tracker is in use

    def get_bbox_tuple(self) -> tuple[float, float, float, float]:
        """Returns the bounding box as a tuple (x1, y1, x2, y2)."""
        return (self.bbox.x1, self.bbox.y1, self.bbox.x2, self.bbox.y2)

    def get_bbox_numpy(self) -> np.ndarray:
        """Returns the bounding box as a NumPy array [x1, y1, x2, y2]."""
        return np.array([self.bbox.x1, self.bbox.y1, self.bbox.x2, self.bbox.y2], dtype=float)

    def get_bbox_xywh(self) -> tuple[float, float, float, float]:
        """Returns the bounding box as a tuple (x, y, width, height)."""
        return (
            self.bbox.x1,
            self.bbox.y1,
            self.bbox.x2 - self.bbox.x1,
            self.bbox.y2 - self.bbox.y1,
        )


def dets_to_xywh(detections: list[Detection]) -> list[list[float]]:
    """Convert Detection objects to XYWH boxes."""
    return [list(detection.get_bbox_xywh()) for detection in detections]

@dataclass
class Frame_Detections:
    """Detections produced for a single video frame."""
    frame_index: int              # 0-indexed frame number in the video
    detections: list[Detection] = field(default_factory=list)

    @property
    def num_detections(self) -> int:
        return len(self.detections)

@dataclass
class TrackingOutput:
    """
    Complete tracker output for a video.
    Includes metadata about the source and camera, plus a Frame_Detections per frame.
    """
    source: str                   # e.g. "yolo_v11m_pt"
    camera_id: str                # e.g. "cam_1"
    fps: float                    # Frames per second of the original video
    frames: list[Frame_Detections] = field(default_factory=list)


def merge_trackings(first: 'TrackingOutput', *rest: 'TrackingOutput') -> 'TrackingOutput':
    """
    Combine N TrackingOutputs frame-by-frame into a new TrackingOutput.

    All inputs must share source, camera_id, frame count, and the per-frame
    frame_index sequence. Detection lists are concatenated per frame, in
    argument order. The returned TrackingOutput is independent of the inputs:
    its frames list and each Frame_Detections.detections list are fresh.

    Parameters:
        first (TrackingOutput): Base output; defines source/camera/fps/frame layout.
        *rest (TrackingOutput): Additional outputs to merge into the result.
    Raises:
        ValueError: If source, camera_id, frame count, or frame indices differ.
    """
    for other in rest:
        if first.source != other.source or first.camera_id != other.camera_id:
            raise ValueError("Can only merge TrackingOutputs from the same source and camera.")
        if len(first.frames) != len(other.frames):
            raise ValueError("Cannot merge TrackingOutputs with different numbers of frames.")
        for frame_first, frame_other in zip(first.frames, other.frames):
            if frame_first.frame_index != frame_other.frame_index:
                raise ValueError("Frame indices do not match between TrackingOutputs.")

    merged_frames: list[Frame_Detections] = []
    for i, frame_first in enumerate(first.frames):
        detections = list(frame_first.detections)
        for other in rest:
            detections.extend(other.frames[i].detections)
        merged_frames.append(Frame_Detections(frame_index=frame_first.frame_index, detections=detections))

    return TrackingOutput(
        source=first.source,
        camera_id=first.camera_id,
        fps=first.fps,
        frames=merged_frames,
    )