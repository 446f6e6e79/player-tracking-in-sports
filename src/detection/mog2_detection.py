import cv2
from tqdm import tqdm

from src.types.tracking import Detection, FrameDetections
from src.types.detection import DetectionOutput, BoundingBox

from src.detection.image_processing import (
    make_clahe,
    make_morph_kernel,
    normalize_illumination,
    opening_closing,
    refine_blobs,
)
from src.utils.logging import get_logger


logger = get_logger(__name__)


def step_mog2(
    mog2: cv2.BackgroundSubtractorMOG2,
    frame: cv2.Mat,
    *,
    clahe: cv2.CLAHE,
    opening_kernel,
    closing_kernel,
    opening_kernel_size: int,
    closing_kernel_size: int,
    min_area: int,
    max_area: int,
    learning_rate: float = -1,
    detect_shadows: bool = False,
) -> cv2.Mat:
    """Run the MOG2 pipeline on a single BGR frame and return the cleaned mask.

    Stateful across calls via the shared `mog2` instance: the caller is
    responsible for constructing it once per stream (see `run_mog2_detection`
    for the canonical setup). Extracted so chunk-based callers (notebook /
    streaming pipelines) can keep the background model alive across chunks.
    """
    norm_frame = normalize_illumination(frame, clahe=clahe)
    mask = mog2.apply(norm_frame, learningRate=learning_rate)

    if detect_shadows:
        mask[mask == 127] = 0

    mask = opening_closing(
        mask, opening_kernel_size, closing_kernel_size,
        opening_kernel=opening_kernel, closing_kernel=closing_kernel,
    )
    return refine_blobs(mask, min_area, max_area)


def run_mog2_detection(
    frames: list[cv2.Mat],
    learning_rate: float = -1,
    history_length: int = 1000,
    var_threshold: float = 25,
    detect_shadows: bool = False,
    opening_kernel_size: int = 7,
    closing_kernel_size: int = 11,
    min_area: int = 1500,
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
        - learning_rate: MOG2 background model learning rate. Default 0.03.
        - history_length: Number of previous frames used for background modeling. Default 1000.
        - var_threshold: MOG2 variance threshold for pixel classification. Default 25.
        - detect_shadows: Whether MOG2 should detect shadows. Default False.
        - opening_kernel_size: Kernel size for morphological opening. Default 7.
        - closing_kernel_size: Kernel size for morphological closing. Default 11.
        - min_area: Minimum blob area in pixels to keep. Default 1500.
        - max_area: Maximum blob area in pixels to keep. Default 200000.
        - heat_up_frames: Number of initial frames to feed into MOG2 without outputting masks, allowing the background model to stabilize. Default 10.
    Returns:
        A list of cleaned binary masks corresponding to each input frame, where white pixels indicate detected motion.
    """

    mog2 = cv2.createBackgroundSubtractorMOG2(
        history=history_length,
        varThreshold=var_threshold,
        detectShadows=detect_shadows,
    )

    # Build the CLAHE object and the morph kernels once — they are stateless
    # across frames and OpenCV reallocates a non-trivial amount per call.
    clahe = make_clahe()
    opening_kernel = make_morph_kernel(opening_kernel_size)
    closing_kernel = make_morph_kernel(closing_kernel_size)

    masks = [
        step_mog2(
            mog2, frame,
            clahe=clahe,
            opening_kernel=opening_kernel,
            closing_kernel=closing_kernel,
            opening_kernel_size=opening_kernel_size,
            closing_kernel_size=closing_kernel_size,
            min_area=min_area,
            max_area=max_area,
            learning_rate=learning_rate,
            detect_shadows=detect_shadows,
        )
        for frame in tqdm(frames, desc="MOG2", unit="frame", leave=False)
    ]

    # Black out the first heat_up_frames to allow MOG2 background model to stabilize without producing noisy masks
    for i in range(min(heat_up_frames, len(masks))):
        masks[i][:] = 0

    return masks


def mog2_frame_to_detections(
    mask: cv2.Mat,
    frame_index: int,
) -> FrameDetections:
    """Convert one MOG2 binary mask into a `FrameDetections` (one box per blob)."""
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    # label 0 is background, skip it
    detections = []
    for label in range(1, num_labels):
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])

        detections.append(Detection(
            bbox=BoundingBox(float(x), float(y), float(x + w), float(y + h)),
            confidence=1.0,
            class_id=None,      # MOG2 does not provide class information
            class_name=None,    # MOG2 does not provide class information
        ))
    return FrameDetections(frame_index=frame_index, detections=detections)


def mog2_to_detection_output(
    raw_masks: list[cv2.Mat],
    camera_id: str,
    fps: float,
    source: str = "mog2",
) -> DetectionOutput:
    """Convert raw MOG2 binary masks into a DetectionOutput.
    One Detection is emitted per connected component (blob); confidence is a constant 1.0
    because MOG2 is a hard binary classifier with no probabilistic score.
    Parameters:
        raw_masks: list returned by run_mog2_detection()
        camera_id: identifier for the camera (e.g. "cam_13")
        fps: video frame rate
        source: label for the detector (e.g. "mog2")
    Returns:
        DetectionOutput with one FrameDetections per input mask.
    """
    frames = [
        mog2_frame_to_detections(mask, frame_index)
        for frame_index, mask in enumerate(raw_masks)
    ]
    return DetectionOutput(
        source=source,
        camera_id=camera_id,
        fps=fps,
        frames=frames,
    )
