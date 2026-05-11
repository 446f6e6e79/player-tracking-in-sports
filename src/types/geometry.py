from dataclasses import dataclass, field

from src.types.labeled import LabeledObject
from src.types.serializable import JsonSerializable


@dataclass
class RectifiedPoint(LabeledObject):
    """A single rectified bbox-center point in pixel coordinates."""
    x: float
    y: float
    confidence: float
    class_id: int
    class_name: str
    track_id: int


@dataclass
class FrameRectifiedPoints:
    """Rectified points produced for a single video frame."""
    frame_index: int
    points: list[RectifiedPoint] = field(default_factory=list)

    @property
    def num_points(self) -> int:
        return len(self.points)


@dataclass
class RectifiedPointsOutput(JsonSerializable):
    """Full rectified-points output for a video — mirrors TrackingOutput's top-level shape."""
    source: str
    camera_id: str
    fps: float
    frames: list[FrameRectifiedPoints] = field(default_factory=list)


@dataclass
class Point3D(LabeledObject):
    """A single 3D point in world coordinates (mm), tagged with its class identity."""
    x: float
    y: float
    z: float
    class_id: int
    class_name: str


@dataclass
class FrameTriangulatedPoints:
    """3D points reconstructed for a single (synchronised) video frame."""
    frame_index: int
    points: list[Point3D] = field(default_factory=list)

    @property
    def num_points(self) -> int:
        return len(self.points)


@dataclass
class TriangulationOutput(JsonSerializable):
    """Full triangulated output across all cameras for one synchronised game act."""
    fps: float
    camera_ids: list[str]
    frames: list[FrameTriangulatedPoints] = field(default_factory=list)
