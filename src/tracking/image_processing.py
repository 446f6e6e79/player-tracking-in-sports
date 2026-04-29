import cv2
import numpy as np

def remove_reflections(
        frames: list[cv2.Mat],
        max_brightness: int = 220,
    ) -> list[cv2.Mat]:
    """
    Reduces specular reflections in a list of BGR frames by clamping the HSV Value channel.
    Pixels brighter than max_brightness are dimmed to max_brightness; all other pixels are untouched.
    Parameters:
        - frames: A list of video frames (in BGR format) to process.
        - max_brightness: Maximum allowed brightness in the Value channel (0-255). Default is 220.
    Returns:
        A list of BGR frames with reflections attenuated.
    """

    processed_frames = []
    for frame in frames:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, max_brightness)  # Clamp the Value channel to reduce reflections
        processed_frames.append(cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR))

    return processed_frames


def opening_closing(
        frames: list[cv2.Mat],
        opening_kernel_size: int = 3,
        closing_kernel_size: int = 3
    ) -> list[cv2.Mat]:
    """
    Applies morphological opening and closing operations to a list of binary masks.
    It uses an elliptical structuring element for both operations, with specified kernel sizes.
    Parameters:
        - frames: A list of binary masks (in BGR format) to process.
        - opening_kernel_size: The size of the kernel for the opening operation. Default is 3.
        - closing_kernel_size: The size of the kernel for the closing operation. Default is 3.
    Returns:
        A list of processed binary masks after applying opening and closing operations.
    """
    
    # Create structuring elements (kernels) for opening and closing operations
    opening_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (opening_kernel_size, opening_kernel_size))
    closing_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (closing_kernel_size, closing_kernel_size))
    
    # Apply opening and closing operations to each frame and store the results
    processed_frames = []
    for frame in frames:
        # Apply morphological opening to remove small noise
        opened = cv2.morphologyEx(frame, cv2.MORPH_OPEN, opening_kernel)
        # Apply morphological closing to fill small holes
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, closing_kernel)
        processed_frames.append(closed)
    
    return processed_frames
