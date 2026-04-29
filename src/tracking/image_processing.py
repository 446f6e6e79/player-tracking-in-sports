import cv2

def opening_closing(
        frames: list[cv2.Mat],
        opening_kernel_size: int = 3,
        closing_kernel_size: int = 3
    ) -> list[cv2.Mat]:
    """
    Applies morphological opening and closing operations to a list of binary masks.
    Parameters:
        - frames: A list of binary masks (in BGR format) to process.
        - opening_kernel_size: The size of the kernel for the opening operation. Default is 3.
        - closing_kernel_size: The size of the kernel for the closing operation. Default is 3.
    Returns:
        A list of processed binary masks after applying opening and closing operations.
    """
    
    # Create structuring elements (kernels) for opening and closing operations
    opening_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (opening_kernel_size, opening_kernel_size))
    closing_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (closing_kernel_size, closing_kernel_size))
    
    # Apply opening and closing operations to each frame and store the results
    processed_frames = []
    for frame in frames:
        # Apply morphological opening to remove small noise
        opened = cv2.morphologyEx(frame, cv2.MORPH_OPEN, opening_kernel)
        # Apply morphological closing to fill small holes
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, closing_kernel)
        processed_frames.append(closed)
    
    return processed_frames
