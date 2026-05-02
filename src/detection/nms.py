"""Class-agnostic NMS over a TrackingOutput.

YOLO's built-in NMS only suppresses overlaps within the same class. Our
fine-tuned model emits identity-encoded classes (`Red_5`, `Red_11`, ...) so
two predictions of the same physical player under different identities both
survive. We collapse them here with a single greedy IoU pass that ignores
class names, run after `merge_trackings` and before `apply_deep_sort`.
"""
from __future__ import annotations

import numpy as np

from src.tracking.deep_sort_components.matching import iou
from src.types.tracking import Detection, Frame_Detections, TrackingOutput


def class_agnostic_nms(
    tracking_output: TrackingOutput,
    iou_threshold: float = 0.5,
) -> TrackingOutput:
    """Greedy class-agnostic NMS, frame by frame.

    Within each frame, sort detections by confidence descending and keep a
    box only if its IoU with every already-kept box is below `iou_threshold`.
    Class names are not consulted — that is the whole point.
    """
    new_frames: list[Frame_Detections] = []
    for fd in tracking_output.frames:
        ordered = sorted(fd.detections, key=lambda d: d.confidence, reverse=True)
        kept: list[Detection] = []
        kept_boxes: list[tuple[float, float, float, float]] = []
        for d in ordered:
            box = d.get_bbox_tuple()
            if kept_boxes:
                ious = iou(np.asarray(box, dtype=float), np.asarray(kept_boxes, dtype=float))
                if ious.max() >= iou_threshold:
                    continue
            kept.append(d)
            kept_boxes.append(box)
        new_frames.append(Frame_Detections(frame_index=fd.frame_index, detections=kept))

    return TrackingOutput(
        source=tracking_output.source,
        camera_id=tracking_output.camera_id,
        fps=tracking_output.fps,
        frames=new_frames,
    )
