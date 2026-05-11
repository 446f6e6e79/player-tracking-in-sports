from dataclasses import dataclass, field

from src.types.detection import Detection, FrameDetections
from src.types.serializable import JsonSerializable


@dataclass
class TrackedDetection(Detection):
    """A detection that has been associated with a track by a tracker."""
    track_id: int


@dataclass
class FrameTrackedDetections:
    """Detections produced for a single video frame, post-tracking."""
    frame_index: int
    detections: list[TrackedDetection] = field(default_factory=list)

    @property
    def num_detections(self) -> int:
        return len(self.detections)


@dataclass
class TrackingOutput(JsonSerializable):
    """Full post-tracking output for a video."""
    source: str
    camera_id: str
    fps: float
    frames: list[FrameTrackedDetections] = field(default_factory=list)
