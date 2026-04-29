from ultralytics import YOLO
import time

def run_yolo_tracking(
    model: YOLO, 
    frames_color: list,
    conf_threshold: float = 0.3,
    tracker_config: str = "botsort.yaml"
) -> list:
    """Run YOLO tracking on a list of color frames and return the tracked results.
    Parameters:
        model (YOLO): The YOLO model for tracking.
        frames_color (list): A list of color frames for tracking.
        conf_threshold (float): The confidence threshold for detections.
        tracker_config (str): The configuration file for the tracker.
    Returns:
        list: A list of tracked results for each frame.
    """

    tracked_results = []
    start_time = time.time()
    
    for i, frame in enumerate(frames_color):
        # Run YOLO tracking with persist=True to maintain track IDs across frames
        result = model.track(
            frame,                  # Input frame for tracking
            persist=True,           # Keep track IDs consistent across frames
            conf=conf_threshold,               # Confidence threshold for detections
            tracker=tracker_config  # Use the specified tracker
        )
        # Append the tracking result for this frame to the list of tracked results
        tracked_results.append(result[0])
    
        # Print progress every 100 frames
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(frames_color)} frames ({(i + 1) / (time.time() - start_time):.1f} fps)")
    
    return tracked_results

def extract_tracking_data(
    tracked_results: list,
    model: YOLO
) -> list:
    """Extract tracking data (bounding boxes, class IDs, confidence scores, and track IDs) from the tracked results.
    Parameters:
        tracked_results (list): A list of tracked results for each frame.
        model (YOLO): The YOLO model for extracting class names.
    Returns:
        list: A list of dictionaries containing tracking data for each frame.
    """
    detection_data_by_frame = []

    # For each frame's tracking result, extract the relevant data 
    for frame_idx, result in enumerate(tracked_results):
        frame_detections = {
            "frame_id": frame_idx,
            "num_detections": len(result.boxes),
            "detections": []
        }

        # For each detected object in the frame    
        for j in range(len(result.boxes)):

            # Extract bounding box coordinates
            bbox = result.boxes.xyxy[j].cpu().numpy() if hasattr(result.boxes.xyxy[j], 'cpu') else result.boxes.xyxy[j]
            
            # Get track ID
            track_id = int(result.boxes.id[j].item()) if result.boxes.id is not None else None
            
            # Get confidence score
            conf = float(result.boxes.conf[j].item())
            
            # Get class ID and name
            class_id = int(result.boxes.cls[j].item())
            class_name = model.names[class_id]
            
            # Build the detection dictionary for this object and append it to the frame's detections list
            detection = {
                "track_id": track_id,
                "bbox": bbox.tolist(), 
                "confidence": round(conf, 3),
                "class_id": class_id,
                "class_name": class_name
            }
            frame_detections["detections"].append(detection)
        
        # Append the frame's detection data to the overall list
        detection_data_by_frame.append(frame_detections)
    return detection_data_by_frame