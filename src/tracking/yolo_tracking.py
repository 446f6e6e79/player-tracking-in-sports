from ultralytics import YOLO
import time

from src.tracking.schema import BoundingBox, Detection, Frame_Detections, TrackingOutput

def run_yolo_tracking(
    model: YOLO,
    frames: list,
    conf_threshold: float = 0.3,
    inference_size: int = 1280,          # Added parameter for flexibility
    class_ids: list[int] | None = None,  # Restrict detection to these class IDs (None = all classes)
) -> list:
    raw_tracked_results = []
    start_time = time.time()

    for i, frame in enumerate(frames):
        result = model.track(
            frame,
            conf=conf_threshold,
            imgsz=inference_size,      # Increase the inference size for better accuracy
            classes=class_ids,         # Filter detections to only the specified class IDs (if provided)
            verbose=False,             # Suppress detailed output for cleaner logs
            persist=True,              # Keep track IDs consistent across frames
            tracker="bytetrack.yaml"   # Use ByteTrack for better tracking performance
        )
        raw_tracked_results.append(result[0])

        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(frames)} frames ({(i + 1) / (time.time() - start_time):.1f} fps)")

    return raw_tracked_results

def yolo_to_tracking_output(
    raw_tracked_results: list,
    model: YOLO,
    camera_id: str,
    fps: float,
    source: str = "yolo",
) -> TrackingOutput:
    """Convert raw YOLO tracking results into a TrackingOutput.
    Parameters:
        raw_tracked_results: list returned by run_yolo_tracking()
        model: the YOLO model (used for class name lookup)
        camera_id: identifier for the camera (e.g. "cam_13")
        fps: video frame rate
        source: label for the tracker (e.g. "yolo_v11m_pt")
    Returns:
        TrackingOutput with one Frame_Detections per input frame.
    """
    # Build the list of Frame_Detections for each frame
    frames = []
    for frame_index, raw_frame_data in enumerate(raw_tracked_results):

        # Build the list of Detection objects for this frame
        detections = []
        for j in range(len(raw_frame_data.boxes)):
            # For each detected box, extract its information
            raw_bbox = raw_frame_data.boxes.xyxy[j].cpu().numpy() if hasattr(raw_frame_data.boxes.xyxy[j], 'cpu') else raw_frame_data.boxes.xyxy[j]
            class_id = int(raw_frame_data.boxes.cls[j].item())
            # Track ID may be missing on the very first frame or for unmatched detections
            track_id = int(raw_frame_data.boxes.id[j].item()) if raw_frame_data.boxes.id is not None else None

            # Build a Detection object and add it to the list for this frame
            detections.append(Detection(
                bbox=BoundingBox(float(raw_bbox[0]), float(raw_bbox[1]), float(raw_bbox[2]), float(raw_bbox[3])),
                confidence=round(float(raw_frame_data.boxes.conf[j].item()), 3),
                class_id=class_id,
                class_name=model.names[class_id],
                track_id=track_id,
            ))
        # Append the Frame_Detections for this frame to the list of frames in the output
        frames.append(Frame_Detections(frame_index=frame_index, detections=detections))

    return TrackingOutput(
        source=source,
        camera_id=camera_id,
        fps=fps,
        frames=frames,
    )
