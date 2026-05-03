import json
from pathlib import Path

from src.types.tracking import (
    BoundingBox,
    FrameTrackedDetections,
    TrackedDetection,
    TrackingOutput,
)


def load_annotations(camera_id: str) -> TrackingOutput:
    """Load ground-truth processed annotations for a single camera from disk.
    Annotations are stored as JSON files matching the post-tracking schema; the
    on-disk file may not include `track_id`, so we synthesize a stable id per
    `class_name` (the canonical identity in our annotations — e.g. "White_14",
    "Red_7"). Eval continues to use class_name for GT identity; the synthetic
    id only exists to satisfy the TrackedDetection contract.
    Parameters:
        - camera_id: camera identifier used to resolve the file name (e.g. "cam_2").
    Returns:
        TrackingOutput populated with ground-truth detections.
    """
    project_root = Path(__file__).parent.parent.parent.parent
    path = project_root / "data" / "annotations" / f"{camera_id}.json"
    with open(path) as f:
        data = json.load(f)

    class_name_to_id: dict[str, int] = {}
    next_id = 1

    frames = []
    for frame in data["frames"]:
        detections = []
        for det in frame["detections"]:
            class_name = det["class_name"]
            # Stable per-identity track_id derived from class_name on first sight.
            if class_name not in class_name_to_id:
                class_name_to_id[class_name] = next_id
                next_id += 1
            track_id = det.get("track_id")
            if track_id is None:
                track_id = class_name_to_id[class_name]

            detections.append(TrackedDetection(
                bbox=BoundingBox(**det["bbox"]),
                confidence=det["confidence"],
                class_id=det["class_id"],
                class_name=class_name,
                track_id=track_id,
            ))
        frames.append(FrameTrackedDetections(frame_index=frame["frame_index"], detections=detections))

    return TrackingOutput(
        source=data["source"],
        camera_id=data["camera_id"],
        fps=data["fps"],
        frames=frames,
    )
