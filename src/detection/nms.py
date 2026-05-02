import numpy as np

from src.utils.iou import iou
from src.types.tracking import Detection, Frame_Detections, TrackingOutput


def class_independent_nms(
    tracking_output: TrackingOutput,
    iou_threshold: float = 0.5,
) -> TrackingOutput:
    """
    Greedy class-independent NMS, frame by frame.
    For each frame, sort the detections by confidence. Iterate through the detections
    in descending confidence order, keeping a detection iff its IoU with all previously kept
    detections is below the `iou_threshold`.
    
    Parameters:
        - tracking_output: The input TrackingOutput containing frames and detections.
        - iou_threshold: IoU threshold for suppressing overlapping detections (default 0.5).
    
    Returns:
        - A new TrackingOutput with the same structure but with detections filtered by NMS.
    """
    new_frames: list[Frame_Detections] = []

    for frame_detections in tracking_output.frames:
        # Sort all detections in the frame by confidence in descending order
        ordered_detections = sorted(frame_detections.detections, key=lambda d: d.confidence, reverse=True)
        
        #List of kept detections
        kept: list[Detection] = []
        # List of bounding boxes of kept detections, used for IoU comparison
        kept_boxes: list[np.ndarray] = []
        
        for detection in ordered_detections:
            # Get the bounding box of the current detection as a numpy array
            box = detection.get_bbox_numpy()
            
            # If there are already kept boxes, compute IoU of the current box with all kept boxes
            if kept_boxes:
                ious = iou(box, np.array(kept_boxes))
                if ious.max() >= iou_threshold:
                    continue
            
            # If we reach here, we keep the detection and add its box to the list of kept boxes
            kept.append(detection)
            kept_boxes.append(box)
        
        new_frames.append(Frame_Detections(frame_index=frame_detections.frame_index, detections=kept))

    return TrackingOutput(
        source=tracking_output.source,
        camera_id=tracking_output.camera_id,
        fps=tracking_output.fps,
        frames=new_frames,
    )
