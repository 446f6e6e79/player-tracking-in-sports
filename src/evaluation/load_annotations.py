import json
from pathlib import Path

from src.tracking.schema import BoundingBox, Detection, Frame_Detections, TrackingOutput


def load_annotations(camera_id: str) -> TrackingOutput:
    """Load ground-truth annotations for a single camera from disk.
    Annotations are stored as JSON files that already match the TrackingOutput schema,
    so this function is a straightforward deserialisation from dict to dataclasses.
    Parameters:
        - camera_id: camera identifier used to resolve the file name (e.g. "cam_2").
    Returns:
        TrackingOutput populated with ground-truth detections.
        Note: track_id is always None in ground-truth files; player identity is
        encoded in class_name (e.g. "White_14", "Red_7").
    """

    # Construct the file path for the specified camera's annotation JSON
    project_root = Path(__file__).parent.parent.parent
    path = project_root / "data" / "annotations" / f"{camera_id}.json"
    with open(path) as f:
        data = json.load(f)

    # Convert the loaded JSON data into TrackingOutput dataclasses
    frames = []
    for frame in data["frames"]:
        detections = []
        for det in frame["detections"]:
            detections.append(Detection(
                bbox=BoundingBox(**det["bbox"]),   # keys x1/y1/x2/y2 match dataclass fields exactly
                confidence=det["confidence"],
                class_id=det["class_id"],
                class_name=det["class_name"],
                track_id=det.get("track_id"),      # null in JSON becomes None
            ))
        frames.append(Frame_Detections(frame_index=frame["frame_index"], detections=detections))

    return TrackingOutput(
        source=data["source"],
        camera_id=data["camera_id"],
        fps=data["fps"],
        frames=frames,
    )
