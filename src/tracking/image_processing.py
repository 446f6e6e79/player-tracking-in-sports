import cv2
import numpy as np

def normalize_illumination(
        frames: list[cv2.Mat],
        clip_limit: float = 2.0,
        tile_grid_size: tuple[int, int] = (8, 8)
    ) -> list[cv2.Mat]:
    """
    Normalizes illumination to reduce shadows and specular reflections.
    Applies CLAHE to the L channel of LAB color space.
    Parameters:
        - frames: A list of video frames (in BGR format).
        - clip_limit: CLAHE contrast clip limit. Higher = stronger correction. Default 2.0.
        - tile_grid_size: Grid size for local histogram computation. Default (8, 8).
    Returns:
        A list of BGR frames with normalized illumination.
    """
    # Create a CLAHE object with the specified clip limit and tile grid size
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    # Process each frame
    processed_frames = []
    for frame in frames:
        # Convert the frame from BGR to LAB color space
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        # Apply CLAHE to the L channel to enhance contrast
        l = clahe.apply(l)

        # Merge the processed L channel back with the original A and B channels, convert back to BGR color space, and store the result
        processed_frames.append(cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR))
        
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
