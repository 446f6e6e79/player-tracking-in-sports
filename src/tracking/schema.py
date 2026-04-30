from dataclasses import dataclass, field

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
