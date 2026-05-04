import json
from pathlib import Path

from src.types.tracking import (
    BoundingBox,
    FrameTrackedDetections,
    TrackedDetection,
    TrackingOutput,
)


def _normalize_frame_indices(raw_frames: list[dict], camera_id: str) -> list[dict]:
    """Return frames with 0-based contiguous indices relative to the first frame.

    Existing annotation files may start at 1 (legacy) while the runtime
    detection/tracking pipeline is 0-based. We normalize once at load time so
    all downstream evaluation code receives the same convention.
    """
    if not raw_frames:
        return raw_frames

    # Validate that frame indices are strictly increasing (no duplicates, no regressions)
    indices = [int(frame["frame_index"]) for frame in raw_frames]
    for prev, curr in zip(indices, indices[1:]):
        if curr <= prev:
            raise ValueError(
                f"Annotation frame_index must be strictly increasing for {camera_id}. "
                f"Found non-increasing pair: {prev}, {curr}."
            )

    # Normalize to 0-based by subtracting the first frame index from all frames.
    offset = indices[0]
    normalized: list[dict] = []
    for frame in raw_frames:
        normalized.append({
            "frame_index": int(frame["frame_index"]) - offset,
            "detections": frame["detections"],
        })
    return normalized


def load_annotations(camera_id: str, version: str = "tracking_01") -> TrackingOutput:
    """Load ground-truth processed annotations for a single camera from disk.
    Annotations are stored as JSON files matching the post-tracking schema; the
    on-disk file may not include `track_id`, so we synthesize a stable id per
    `class_name` (the canonical identity in our annotations — e.g. "White_14",
    "Red_7"). Eval continues to use class_name for GT identity; the synthetic
    id only exists to satisfy the TrackedDetection contract.
    Parameters:
        - camera_id: camera identifier used to resolve the file name (e.g. "cam_2").
        - version: annotation version subdirectory under data/annotations/ (default "tracking_01").
    Returns:
        TrackingOutput populated with ground-truth detections.
        Frame indices are normalized to 0-based at load time.
    """
    project_root = Path(__file__).parent.parent.parent.parent
    path = project_root / "data" / "annotations" / version / f"{camera_id}.json"
    with open(path) as f:
        data = json.load(f)

    raw_frames = _normalize_frame_indices(data["frames"], camera_id)

    class_name_to_id: dict[str, int] = {}
    next_id = 1

    frames = []
    for frame in raw_frames:
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
