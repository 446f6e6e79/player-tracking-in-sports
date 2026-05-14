import warnings

import numpy as np

from src.utils.iou import iou
from src.types.tracking import Detection, FrameDetections
from src.types.detection import DetectionOutput


def class_independent_nms(
    detection_output: DetectionOutput,
    iou_threshold: float = 0.5,
) -> DetectionOutput:
    """
    Greedy class-independent NMS, frame by frame.
    For each frame, sort the detections by confidence. Iterate through the detections
    in descending confidence order, keeping a detection iff its IoU with all previously kept
    detections is below the `iou_threshold`.

    Parameters:
        - detection_output: The input DetectionOutput containing frames and detections.
        - iou_threshold: IoU threshold for suppressing overlapping detections (default 0.5).

    Returns:
        - A new DetectionOutput with the same structure but with detections filtered by NMS.
    """
    if not detection_output.frames:
        warnings.warn(
            "class_independent_nms received a DetectionOutput with no frames; "
            "returning it unchanged.",
            stacklevel=2,
        )
        return detection_output

    new_frames: list[FrameDetections] = []

    for frame_detections in detection_output.frames:
        # Sort all detections in the frame by confidence in descending order
        ordered_detections = sorted(frame_detections.detections, key=lambda d: d.confidence, reverse=True)

        kept: list[Detection] = []
        kept_boxes: list[np.ndarray] = []

        for detection in ordered_detections:
            box = detection.get_bbox_numpy()

            if kept_boxes:
                ious = iou(box, np.array(kept_boxes))
                if ious.max() >= iou_threshold:
                    continue

            kept.append(detection)
            kept_boxes.append(box)

        new_frames.append(FrameDetections(frame_index=frame_detections.frame_index, detections=kept))

    return detection_output.with_frames(new_frames)
