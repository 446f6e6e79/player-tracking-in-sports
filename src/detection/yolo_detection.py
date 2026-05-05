from ultralytics import YOLO
import time

from src.types.tracking import BoundingBox, Detection, DetectionOutput, FrameDetections


def run_yolo_detection(
    model: YOLO,
    frames: list,
    conf_threshold: float = 0.3,
    inference_size: int = 640,
    iou_threshold: float = 0.45,
    class_ids: list[int] | None = None,  # Restrict detection to these class IDs (None = all classes)
) -> list:
    """Run YOLO detection on a list of frames.
    Parameters:
        model: the YOLO model to use for detection
        frames: list of frames (as numpy arrays) to run detection on
        conf_threshold: confidence threshold for filtering detections (default 0.3)
        inference_size: size to which frames are resized for inference (default 640)
        iou_threshold: IoU threshold for non-max suppression (default 0.45)
        class_ids: list of class IDs to detect (default None = detect all classes)
    Returns:
        List of raw YOLO results (one per frame) as returned by model.predict()
    """
    raw_results = []
    start_time = time.time()

    for i, frame in enumerate(frames):
        result = model.predict(
            frame,
            conf=conf_threshold,
            imgsz=inference_size,      # Increase the inference size for better accuracy
            iou=iou_threshold,         # Set IoU threshold for NMS
            classes=class_ids,         # Filter detections to only the specified class IDs (if provided)
            verbose=False,             # Suppress detailed output for cleaner logs
        )
        raw_results.append(result[0])

        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(frames)} frames ({(i + 1) / (time.time() - start_time):.1f} fps)")

    return raw_results


def yolo_to_detection_output(
    raw_results: list,
    model: YOLO,
    camera_id: str,
    fps: float,
    source: str = "yolo",
) -> DetectionOutput:
    """Convert raw YOLO detection results into a DetectionOutput (pre-tracking).
    Parameters:
        raw_results: list returned by run_yolo_detection()
        model: the YOLO model (used for class name lookup)
        camera_id: identifier for the camera (e.g. "cam_13")
        fps: video frame rate
        source: label for the detector (e.g. "yolo_v11m_pt")
    Returns:
        DetectionOutput with one FrameDetections per input frame.
    """
    frames = []
    for frame_index, raw_frame_data in enumerate(raw_results):

        # Build the list of Detection objects for this frame
        detections = []
        for j in range(len(raw_frame_data.boxes)):
            # For each detected box, extract its information
            raw_bbox = raw_frame_data.boxes.xyxy[j].cpu().numpy() if hasattr(raw_frame_data.boxes.xyxy[j], 'cpu') else raw_frame_data.boxes.xyxy[j]
            class_id = int(raw_frame_data.boxes.cls[j].item())

            # Build a Detection object and add it to the list for this frame
            detections.append(Detection(
                bbox=BoundingBox(float(raw_bbox[0]), float(raw_bbox[1]), float(raw_bbox[2]), float(raw_bbox[3])),
                confidence=round(float(raw_frame_data.boxes.conf[j].item()), 3),
                class_id=class_id,
                class_name=model.names[class_id],
            ))
        frames.append(FrameDetections(frame_index=frame_index, detections=detections))

    return DetectionOutput(
        source=source,
        camera_id=camera_id,
        fps=fps,
        frames=frames,
    )
