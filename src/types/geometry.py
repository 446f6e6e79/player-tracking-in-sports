from dataclasses import dataclass, field


@dataclass
class RectifiedPoint:
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
class RectifiedPointsOutput:
    """Full rectified-points output for a video — mirrors TrackingOutput's top-level shape."""
    source: str
    camera_id: str
    fps: float
    frames: list[FrameRectifiedPoints] = field(default_factory=list)
