import json
import os
from dataclasses import dataclass, field

import numpy as np


@dataclass
class BoundingBox:
    """Represents a bounding box in pixel coordinates."""
    x1: float
    y1: float
    x2: float
    y2: float

    def get_center(self) -> tuple[float, float]:
        """Returns the geometric center of the bounding box as (cx, cy)."""
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)


@dataclass
class Detection:
    """A single object detection in a frame, before tracking has run."""
    bbox: BoundingBox             # Bounding box coordinates (x1, y1, x2, y2)
    confidence: float             # Confidence score of the detection (0.0 to 1.0)
    class_id: int                 # Class ID of the detection
    class_name: str               # Human-readable class name (e.g. "player")

    def get_bbox_tuple(self) -> tuple[float, float, float, float]:
        """Returns the bounding box as a tuple (x1, y1, x2, y2)."""
        return (self.bbox.x1, self.bbox.y1, self.bbox.x2, self.bbox.y2)

    def get_int_bbox_tuple(self) -> tuple[int, int, int, int]:
        """Returns the bounding box as a tuple of integers (x1, y1, x2, y2)."""
        return tuple(map(int, self.get_bbox_tuple()))

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


@dataclass
class TrackedDetection(Detection):
    """A detection that has been associated with a track by a tracker."""
    track_id: int


def dets_to_xywh(detections: list[Detection]) -> list[list[float]]:
    """Convert Detection objects to XYWH boxes."""
    return [list(detection.get_bbox_xywh()) for detection in detections]


@dataclass
class FrameDetections:
    """Detections produced for a single video frame, pre-tracking."""
    frame_index: int              # 0-indexed frame number in the video
    detections: list[Detection] = field(default_factory=list)

    @property
    def num_detections(self) -> int:
        return len(self.detections)


@dataclass
class FrameTrackedDetections:
    """Detections produced for a single video frame, post-tracking."""
    frame_index: int
    detections: list[TrackedDetection] = field(default_factory=list)

    @property
    def num_detections(self) -> int:
        return len(self.detections)


@dataclass
class DetectionOutput:
    """Full pre-tracking detector output for a video."""
    source: str                   # e.g. "yolo_v11m_pt"
    camera_id: str                # e.g. "cam_1"
    fps: float                    # Frames per second of the original video
    frames: list[FrameDetections] = field(default_factory=list)


@dataclass
class TrackingOutput:
    """Full post-tracking output for a video."""
    source: str
    camera_id: str
    fps: float
    frames: list[FrameTrackedDetections] = field(default_factory=list)

    def write(self, path: str, overwrite: bool = False) -> None:
        """Serialize this TrackingOutput to a JSON file.
        Parameters:
            - path (str): Destination file path.
            - overwrite (bool): If False (default) and `path` already exists,
              raises FileExistsError instead of overwriting.
        """
        if not overwrite and os.path.exists(path):
            raise FileExistsError(
                f"Refusing to overwrite existing file: {path}. Pass overwrite=True to replace it."
            )

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

        payload = {
            "source": self.source,
            "camera_id": self.camera_id,
            "fps": self.fps,
            "frames": [
                {
                    "frame_index": frame.frame_index,
                    "detections": [
                        {
                            "bbox": {
                                "x1": det.bbox.x1,
                                "y1": det.bbox.y1,
                                "x2": det.bbox.x2,
                                "y2": det.bbox.y2,
                            },
                            "confidence": det.confidence,
                            "class_id": det.class_id,
                            "class_name": det.class_name,
                            "track_id": det.track_id,
                        }
                        for det in frame.detections
                    ],
                }
                for frame in self.frames
            ],
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f)

    @classmethod
    def read(cls, path: str) -> "TrackingOutput":
        """Load a TrackingOutput from a JSON file written by `write`.
        Parameters:
            - path (str): Source file path.
        Returns:
            - TrackingOutput: Reconstructed instance with all detections, including track_ids.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"TrackingOutput file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        frames = [
            FrameTrackedDetections(
                frame_index=frame["frame_index"],
                detections=[
                    TrackedDetection(
                        bbox=BoundingBox(**det["bbox"]),
                        confidence=det["confidence"],
                        class_id=det["class_id"],
                        class_name=det["class_name"],
                        track_id=det["track_id"],
                    )
                    for det in frame["detections"]
                ],
            )
            for frame in data["frames"]
        ]

        return cls(
            source=data["source"],
            camera_id=data["camera_id"],
            fps=data["fps"],
            frames=frames,
        )


def merge_detections(first: 'DetectionOutput', *rest: 'DetectionOutput') -> 'DetectionOutput':
    """
    Combine N DetectionOutputs frame-by-frame into a new DetectionOutput.

    All inputs must share source, camera_id, frame count, and the per-frame
    frame_index sequence. Detection lists are concatenated per frame, in
    argument order. The returned DetectionOutput is independent of the inputs:
    its frames list and each FrameDetections.detections list are fresh.

    Parameters:
        first (DetectionOutput): Base output; defines source/camera/fps/frame layout.
        *rest (DetectionOutput): Additional outputs to merge into the result.
    Raises:
        ValueError: If source, camera_id, frame count, or frame indices differ.
    """
    for other in rest:
        if first.source != other.source or first.camera_id != other.camera_id:
            raise ValueError("Can only merge DetectionOutputs from the same source and camera.")
        if len(first.frames) != len(other.frames):
            raise ValueError("Cannot merge DetectionOutputs with different numbers of frames.")
        for frame_first, frame_other in zip(first.frames, other.frames):
            if frame_first.frame_index != frame_other.frame_index:
                raise ValueError("Frame indices do not match between DetectionOutputs.")

    merged_frames: list[FrameDetections] = []
    for i, frame_first in enumerate(first.frames):
        detections = list(frame_first.detections)
        for other in rest:
            detections.extend(other.frames[i].detections)
        merged_frames.append(FrameDetections(frame_index=frame_first.frame_index, detections=detections))

    return DetectionOutput(
        source=first.source,
        camera_id=first.camera_id,
        fps=first.fps,
        frames=merged_frames,
    )
