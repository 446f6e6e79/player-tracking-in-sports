import cv2

def MOG2_motion_detection(
    frames: list[cv2.Mat],
    learning_rate: float = -1,
    history: int = 300,
    var_threshold: float = 20,
    detect_shadows: bool = True,
):
    """
    Applies the MOG2 (Mixture of Gaussians version 2) motion detection algorithm to a list of video frames.
    Parameters:
        - frames: A list of video frames (in BGR format) to process for motion detection.
        - learning_rate: The learning rate for the background model update. Default is -1 (automatic).
        - history: The number of previous frames to use for background modeling. Default is 500.
        - var_threshold: The variance threshold for the pixel classification. Default is 16.
        - detect_shadows: Whether to detect shadows in the foreground mask. Default is True.
    Returns:
        A list of binary masks corresponding to each input frame, where white pixels indicate detected motion.
    """

    # Create the MOG2 background subtractor with the specified parameters
    mog2 = cv2.createBackgroundSubtractorMOG2(
        history=history,
        varThreshold=var_threshold,
        detectShadows=detect_shadows
    )

    # Apply the MOG2 algorithm to each frame and store the resulting foreground masks in a list
    masks = []
    for frame in frames:
        mask = mog2.apply(frame, learningRate=learning_rate)

        # Remove shadows if detect_shadows is True (shadows are setted to 127)
        if detect_shadows:
            mask[mask == 127] = 0  # Set shadow pixels to black (0)
        
        masks.append(mask)


    return masks