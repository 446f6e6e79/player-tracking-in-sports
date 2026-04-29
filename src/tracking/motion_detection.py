import cv2

def MOG2_motion_detection(
    frames: list[cv2.Mat],
    learning_rate: float = -1,
    history_length: int = 300,
    var_threshold: float = 20,
    detect_shadows: bool = True,
):
    """
    Applies the MOG2 (Mixture of Gaussians version 2) motion detection algorithm to a list of video frames.
    Parameters:
        - frames: A list of video frames (in BGR format) to process for motion detection.
        - learning_rate: The learning rate for the background model update. Default is -1 (automatic).
        - history_length: The number of previous frames to use for background modeling. Default is 300.
        - var_threshold: The variance threshold for the pixel classification. Default is 20.
        - detect_shadows: Whether to detect shadows in the foreground mask. Default is True.
    Returns:
        A list of binary masks corresponding to each input frame, where white pixels indicate detected motion.
    """

    # Create the MOG2 background subtractor with the specified parameters
    mog2 = cv2.createBackgroundSubtractorMOG2(
        history=history_length,
        varThreshold=var_threshold,
        detectShadows=detect_shadows
    )

    # Apply the MOG2 algorithm to each frame and store the resulting foreground masks in a list
    masks = []
    for frame in frames:
        mask = mog2.apply(frame, learningRate=learning_rate)

        # Remove shadows if detect_shadows is True 
        if detect_shadows:
            mask[mask == 127] = 0  # Set shadow pixels to black (0)
        
        masks.append(mask)

    return masks

def refine_blobs(
    masks: list[cv2.Mat],
    min_area: int = 200,
    max_area: int = 5000,
) -> list[cv2.Mat]:
    """
    Filters connected components (blobs) in a list of binary masks by area.
    Blobs outside the [min_area, max_area] range are removed, keeping only
    components that plausibly correspond to player-sized regions.
    Parameters:
        - masks: A list of binary masks to process.
        - min_area: Minimum blob area in pixels to keep. Default is 200.
        - max_area: Maximum blob area in pixels to keep. Default is 5000.
    Returns:
        A list of refined binary masks with only valid blobs retained.
    """

    refined_masks = []
    for mask in masks:
        # Find connected components in the binary mask
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

        # Rebuild mask keeping only blobs within the area range (label 0 is background)
        refined = mask.copy()
        refined[:] = 0
        for label in range(1, num_labels):
            area = stats[label, cv2.CC_STAT_AREA]
            if min_area <= area <= max_area:
                refined[labels == label] = 255

        refined_masks.append(refined)

    return refined_masks