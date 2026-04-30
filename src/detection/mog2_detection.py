import cv2
import numpy as np
import time

from src.detection.schema import BoundingBox, Detection, Frame_Detections, DetectionOutput
from src.utils.image_processing import (
    normalize_illumination,
    remove_field_pixels,
    opening_closing,
    refine_blobs,
)

def run_mog2_detection(
    frames: list[cv2.Mat],
    learning_rate: float = -1,
    history_length: int = 1000,
    var_threshold: float = 15,
    detect_shadows: bool = False,
    opening_kernel_size: int = 7,
    closing_kernel_size: int = 11,
    min_area: int = 1000,
    max_area: int = 200000,
    heat_up_frames: int = 10,
) -> list[cv2.Mat]:
    """
    End-to-end MOG2 motion detection pipeline.

    The pipeline runs in a single per-frame loop:
    illumination normalization → MOG2 background subtraction → field-color suppression →
    morphological opening/closing → blob area filtering.
    Parameters:
        - frames: A list of video frames (in BGR format) to process for motion detection.
        - learning_rate: MOG2 background model learning rate. Default -1 (automatic).
        - history_length: Number of previous frames used for background modeling. Default 1000.
        - var_threshold: MOG2 variance threshold for pixel classification. Default 15.
        - detect_shadows: Whether MOG2 should detect shadows. Default False.
        - opening_kernel_size: Kernel size for morphological opening. Default 7.
        - closing_kernel_size: Kernel size for morphological closing. Default 11.
        - min_area: Minimum blob area in pixels to keep. Default 500.
        - max_area: Maximum blob area in pixels to keep. Default 200000.
        - heat_up_frames: Number of initial frames to feed into MOG2 without outputting masks, allowing the background model to stabilize. Default 10.
    Returns:
        A list of cleaned binary masks corresponding to each input frame, where white pixels indicate detected motion.
    """

    # Build all stateful objects once outside the loop (CLAHE, MOG2 subtractor and
    # structuring elements should not be recreated on each frame)
    mog2 = cv2.createBackgroundSubtractorMOG2(
        history=history_length,
        varThreshold=var_threshold,
        detectShadows=detect_shadows,
    )

    masks = []
    start_time = time.time()
    for i, frame in enumerate(frames):
        # Normalize illumination before MOG2 to mitigate lighting changes, which can cause false positives and missed detections.
        norm_frame = normalize_illumination(frame)

        # MOG2 foreground mask extraction
        mask = mog2.apply(norm_frame, learningRate=learning_rate)

        # During the heat-up period, feed frames into MOG2 to build the background model but skip cleaning/output
        if i < heat_up_frames:
            masks.append(np.zeros_like(mask))

        # Remove shadows if detect_shadows is True
        if detect_shadows:
            mask[mask == 127] = 0  # Set shadow pixels to black (0)

        # Suppress foreground pixels whose underlying color matches the floor, which helps eliminate shadow blobs on reflective floors
        mask = remove_field_pixels(mask, frame)

        # Morphological opening and closing to clean up the mask
        mask = opening_closing(mask, opening_kernel_size, closing_kernel_size)

        # Blob filtering by area to remove small noise and large non-player blobs
        mask = refine_blobs(mask, min_area, max_area)

        masks.append(mask)

        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(frames)} frames ({(i + 1) / (time.time() - start_time):.1f} fps)")

    return masks

def mog2_to_detection_output(
    raw_masks: list[cv2.Mat],
    camera_id: str,
    fps: float,
    source: str = "mog2",
    class_id: int = 0,
    class_name: str = "person",
) -> DetectionOutput:
    """Convert raw MOG2 binary masks into a DetectionOutput.
    One Detection is emitted per connected component (blob); confidence is a constant 1.0
    because MOG2 is a hard binary classifier with no probabilistic score.
    Parameters:
        raw_masks: list returned by run_mog2_detection()
        camera_id: identifier for the camera (e.g. "cam_13")
        fps: video frame rate
        source: label for the detector (e.g. "mog2")
        class_id: class ID assigned to every detection (MOG2 is class-agnostic). Default 0.
        class_name: human-readable class name assigned to every detection. Default "person".
    Returns:
        DetectionOutput with one Frame_Detections per input mask.
    """
    # Build the list of FrameDetections for each frame
    frames = []
    for frame_index, mask in enumerate(raw_masks):

        # Extract one bounding box per connected component in the binary mask
        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

        # Build the list of Detection objects for this frame (label 0 is background, skip it)
        detections = []
        for label in range(1, num_labels):
            x = int(stats[label, cv2.CC_STAT_LEFT])
            y = int(stats[label, cv2.CC_STAT_TOP])
            w = int(stats[label, cv2.CC_STAT_WIDTH])
            h = int(stats[label, cv2.CC_STAT_HEIGHT])

            # Build a Detection object and add it to the list for this frame
            detections.append(Detection(
                bbox=BoundingBox(float(x), float(y), float(x + w), float(y + h)),
                confidence=1.0,
                class_id=class_id,
                class_name=class_name,
            ))
        # Append the FrameDetections for this frame to the list of frames in the output
        frames.append(Frame_Detections(frame_index=frame_index, detections=detections))

    return DetectionOutput(
        source=source,
        camera_id=camera_id,
        fps=fps,
        frames=frames,
    )
