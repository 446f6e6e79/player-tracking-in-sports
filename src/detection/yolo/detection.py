from ultralytics import YOLO
from tqdm import tqdm

from src.types.tracking import Detection, FrameDetections
from src.types.detection import DetectionOutput, BoundingBox
from src.utils.logging import get_logger


logger = get_logger(__name__)


def run_yolo_detection(
    model: YOLO,
    frames: list,
    conf_threshold: float = 0.3,
    inference_size: int = 640,
    iou_threshold: float = 0.45,
    class_ids: list[int] | None = None,  # Restrict detection to these class IDs (None = all classes)
    batch_size: int = 4,
) -> list:
    """Run YOLO detection on a list of frames.
    Parameters:
        model: the YOLO model to use for detection
        frames: list of frames (as numpy arrays) to run detection on
        conf_threshold: confidence threshold for filtering detections (default 0.3)
        inference_size: size to which frames are resized for inference (default 640)
        iou_threshold: IoU threshold for non-max suppression (default 0.45)
        class_ids: list of class IDs to detect (default None = detect all classes)
        batch_size: how many frames to feed `model.predict` per call. Higher
            values amortise GPU launch overhead at the cost of more VRAM/RAM —
            default 4 stays comfortable on laptops.
    Returns:
        List of raw YOLO results (one per frame), in the same order as `frames`.
    """
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    if not frames:
        raise ValueError("run_yolo_detection received an empty frame list.")

    raw_results: list = []
    total = len(frames)

    progress = tqdm(total=total, desc="YOLO", unit="frame", leave=False)
    try:
        for start in range(0, total, batch_size):
            batch = frames[start:start + batch_size]
            results = model.predict(
                batch,
                conf=conf_threshold,
                imgsz=inference_size,      # Increase the inference size for better accuracy
                iou=iou_threshold,         # Set IoU threshold for NMS
                classes=class_ids,         # Filter detections to only the specified class IDs (if provided)
                verbose=False,             # Suppress detailed output for cleaner logs
            )
            raw_results.extend(results)
            progress.update(len(batch))
    finally:
        progress.close()

    return raw_results


def _to_numpy(tensor_like):
    """Best-effort conversion of an Ultralytics tensor to a NumPy array on CPU."""
    return tensor_like.cpu().numpy() if hasattr(tensor_like, "cpu") else tensor_like


def yolo_to_detection_output(
    raw_results: list,
    model: YOLO,
    camera_id: str,
    fps: float,
    source: str = "yolo",
    frame_index_offset: int = 0,
) -> DetectionOutput:
    """Convert raw YOLO detection results into a DetectionOutput (pre-tracking).
    Parameters:
        raw_results: list returned by run_yolo_detection()
        model: the YOLO model (used for class name lookup)
        camera_id: identifier for the camera (e.g. "cam_13")
        fps: video frame rate
        source: label for the detector (e.g. "yolo_v11m_pt")
        frame_index_offset: absolute frame index of the first element in
            `raw_results`. Lets chunked callers reconstruct global indices.
    Returns:
        DetectionOutput with one FrameDetections per input frame.
    """
    frames = []
    class_names = model.names
    for local_index, raw_frame_data in enumerate(raw_results):
        frame_index = local_index + frame_index_offset
        boxes = raw_frame_data.boxes
        # Pull each tensor across to CPU exactly once per frame instead of
        # per-detection — saves a lot of host/device round-trips on dense frames.
        xyxy = _to_numpy(boxes.xyxy)
        confs = _to_numpy(boxes.conf)
        cls_ids = _to_numpy(boxes.cls).astype(int)

        detections = [
            Detection(
                bbox=BoundingBox(float(xyxy[j, 0]), float(xyxy[j, 1]),
                                 float(xyxy[j, 2]), float(xyxy[j, 3])),
                confidence=round(float(confs[j]), 3),
                class_id=int(cls_ids[j]),
                class_name=class_names[int(cls_ids[j])],
            )
            for j in range(len(boxes))
        ]
        frames.append(FrameDetections(frame_index=frame_index, detections=detections))

    return DetectionOutput(
        source=source,
        camera_id=camera_id,
        fps=fps,
        frames=frames,
    )
