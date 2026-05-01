"""Drivers wiring YOLO detections through the DeepSORT-style tracker.

Two entry points:
    apply_deep_sort      — given a detection-only TrackingOutput, return a
                           tracked one. This is what the notebook calls
                           after merging the two-pass (player + ball)
                           detection results.
    run_yolo_deep_sort   — convenience wrapper for the case where you want
                           detection + tracking in one call. Mirrors the
                           shape of run_yolo_tracking() but uses
                           model.predict() — DeepSORT owns the association.
"""
from __future__ import annotations

import time
from typing import Any

from ultralytics import YOLO

from src.tracking.deep_sort import DeepSortTracker
from src.types.tracking import (
    BoundingBox,
    Detection,
    Frame_Detections,
    TrackingOutput,
)


def apply_deep_sort(
    tracking_output: TrackingOutput,
    *,
    max_iou_distance: float = 0.7,
    max_age: int = 30,
    n_init: int = 3,
) -> TrackingOutput:
    """Run a fresh DeepSortTracker over a detection-only TrackingOutput.

    Frames must be in monotonic frame_index order (we sort defensively).
    Existing track_ids on the input are ignored — DeepSORT re-assigns them.
    Unmatched detections are dropped.
    """
    tracker = DeepSortTracker(
        max_iou_distance=max_iou_distance,
        max_age=max_age,
        n_init=n_init,
    )
    new_frames: list[Frame_Detections] = []
    for fd in sorted(tracking_output.frames, key=lambda f: f.frame_index):
        tracked = tracker.update(list(fd.detections))
        new_frames.append(Frame_Detections(frame_index=fd.frame_index, detections=tracked))

    return TrackingOutput(
        source=tracking_output.source,
        camera_id=tracking_output.camera_id,
        fps=tracking_output.fps,
        frames=new_frames,
    )


def run_yolo_deep_sort(
    model: YOLO,
    frames: list,
    camera_id: str,
    fps: float,
    conf_threshold: float = 0.25,
    inference_size: int = 640,
    class_ids: list[int] | None = None,
    source: str = "yolo_deep_sort",
    *,
    max_iou_distance: float = 0.7,
    max_age: int = 30,
    n_init: int = 3,
) -> TrackingOutput:
    """YOLO predict() per frame piped through DeepSortTracker.

    Mirrors run_yolo_tracking()'s signature but uses predict(): we don't
    want Ultralytics' built-in tracker — the whole point is to replace it.
    """
    tracker = DeepSortTracker(
        max_iou_distance=max_iou_distance,
        max_age=max_age,
        n_init=n_init,
    )
    out_frames: list[Frame_Detections] = []
    start_time = time.time()

    for i, frame in enumerate(frames):
        result = model.predict(
            frame,
            conf=conf_threshold,
            imgsz=inference_size,
            classes=class_ids,
            verbose=False,
        )[0]
        per_frame_dets = _yolo_result_to_detections(result, model.names)
        tracked = tracker.update(per_frame_dets)
        out_frames.append(Frame_Detections(frame_index=i, detections=tracked))

        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(frames)} frames ({(i + 1) / (time.time() - start_time):.1f} fps)")

    return TrackingOutput(
        source=source,
        camera_id=camera_id,
        fps=fps,
        frames=out_frames,
    )


def _yolo_result_to_detections(result: Any, names: dict) -> list[Detection]:
    """Per-frame extraction mirroring yolo_tracking.yolo_to_tracking_output,
    minus the track_id field — DeepSortTracker assigns it."""
    detections: list[Detection] = []
    for j in range(len(result.boxes)):
        raw_bbox = (
            result.boxes.xyxy[j].cpu().numpy()
            if hasattr(result.boxes.xyxy[j], "cpu")
            else result.boxes.xyxy[j]
        )
        class_id = int(result.boxes.cls[j].item())
        detections.append(Detection(
            bbox=BoundingBox(
                float(raw_bbox[0]), float(raw_bbox[1]),
                float(raw_bbox[2]), float(raw_bbox[3]),
            ),
            confidence=round(float(result.boxes.conf[j].item()), 3),
            class_id=class_id,
            class_name=names[class_id],
        ))
    return detections
