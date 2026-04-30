from ultralytics import YOLO
import time

from src.detection.schema import BoundingBox, Detection, Frame_Detections, DetectionOutput

def run_yolo_detection(
    model: YOLO,
    frames: list,
    conf_threshold: float = 0.3,
) -> list:
    """Run YOLO detection on a list of color frames and return the raw results.
    Parameters:
        model (YOLO): The YOLO model used for detection.
        frames (list): A list of frames to detect on.
        conf_threshold (float): The confidence threshold for detections.
    Returns:
        list: A list of YOLO result objects, one per frame.
    """

    raw_detected_results = []
    start_time = time.time()

    for i, frame in enumerate(frames):
        # Pure detection: no temporal association, no track IDs
        result = model.predict(
            frame,
            conf=conf_threshold,
            verbose=False,
        )
        raw_detected_results.append(result[0])

        # Print progress every 100 frames
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(frames)} frames ({(i + 1) / (time.time() - start_time):.1f} fps)")

    return raw_detected_results

def yolo_to_detection_output(
    raw_detected_results: list,
    model: YOLO,
    camera_id: str,
    fps: float,
    frame_width: int,
    frame_height: int,
    source: str = "yolo",
) -> DetectionOutput:
    """Convert raw YOLO detection results into a DetectionOutput.
    Parameters:
        raw_detected_results: list returned by run_yolo_detection()
        model: the YOLO model (used for class name lookup)
        camera_id: identifier for the camera (e.g. "cam_13")
        fps: video frame rate
        frame_width: frame width in pixels
        frame_height: frame height in pixels
        source: label for the detector (e.g. "yolo_v8m_pt")
    Returns:
        DetectionOutput with one FrameDetections per input frame.
    """
    # Build the list of FrameDetections for each frame
    frames = []
    for frame_index, raw_frame_data in enumerate(raw_detected_results):
        
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
        # Append the FrameDetections for this frame to the list of frames in the output
        frames.append(Frame_Detections(frame_index=frame_index, detections=detections))

    return DetectionOutput(
        source=source,
        camera_id=camera_id,
        fps=fps,
        frame_width=frame_width,
        frame_height=frame_height,
        frames=frames,
    )
