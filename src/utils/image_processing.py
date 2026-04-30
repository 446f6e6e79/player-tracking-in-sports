import cv2

def normalize_illumination(
        frame: cv2.Mat,
        clip_limit: float = 6.0,
        tile_grid_size: tuple[int, int] = (8, 8)
    ) -> cv2.Mat:
    """
    Single-frame variant of normalize_illumination.
    Takes a precomputed CLAHE object so the same instance can be reused across many frames in a fused pipeline.
    Parameters:
        - frame: A single video frame in BGR format.
        - clahe: A precomputed cv2.CLAHE instance (built with cv2.createCLAHE).
    Returns:
        A BGR frame with normalized illumination.
    """
    # Create a CLAHE object with the specified clip limit and tile grid size
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    # Convert the frame from BGR to LAB color space
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # Apply CLAHE to the L channel to enhance contrast
    l = clahe.apply(l)

    # Merge the processed L channel back with the original A and B channels, convert back to BGR color space
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

def remove_field_pixels(
        mask: cv2.Mat,
        frame: cv2.Mat,
        hue_min: int = 5,
        hue_max: int = 40,
        sat_min: int = 10,
        val_min: int = 30,
    ) -> cv2.Mat:
    """
    Single-frame variant of remove_field_pixels. 
    Zeros mask pixels whose underlying BGR pixel falls in the floor-color HSV range.
    The default HSV range is tuned for a typical indoor parquet floor, but can be adjusted for other surfaces.
    Parameters:
        - mask: Binary motion mask to filter.
        - frame: The corresponding original BGR frame.
        - hue_min: Minimum hue for the floor color (OpenCV 0-179). Default 5.
        - hue_max: Maximum hue for the floor color (OpenCV 0-179). Default 40.
        - sat_min: Minimum saturation. Excludes desaturated white/grey court markings. Default 10.
        - val_min: Minimum brightness. Excludes genuinely dark shadow pixels we want to keep. Default 30.
    Returns:
        A filtered binary mask.
    """
    # Convert frame to HSV and split channels for per-pixel hue/sat/val tests
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # Pixels whose color falls in the floor range are zeroed out of the mask
    field_mask = (h >= hue_min) & (h <= hue_max) & (s >= sat_min) & (v >= val_min)
    out = mask.copy()
    out[field_mask] = 0
    return out

def opening_closing(
        mask: cv2.Mat,
        opening_kernel_size: int,
        closing_kernel_size: int,
    ) -> cv2.Mat:
    """
    Single-frame variant of opening_closing. Takes precomputed structuring elements so the
    same kernels can be reused across many frames in a fused pipeline.
    Parameters:
        - mask: A single binary mask.
        - opening_kernel_size: Kernel size for morphological opening.
        - closing_kernel_size: Kernel size for morphological closing. 
    Returns:
        A binary mask with opening then closing applied.
    """
    # Create structuring elements (kernels) for opening and closing operations
    opening_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (opening_kernel_size, opening_kernel_size))
    closing_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (closing_kernel_size, closing_kernel_size))

    # Apply morphological opening to remove small noise, then closing to fill small holes
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, opening_kernel)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, closing_kernel)
    return closed


def refine_blobs(
    mask: cv2.Mat,
    min_area: int = 500,
    max_area: int = 10000,
) -> cv2.Mat:
    """
    Single-frame variant of refine_blobs. Filters connected components by area in one mask.
    Parameters:
        - mask: A single binary mask.
        - min_area: Minimum blob area in pixels to keep. Default is 500.
        - max_area: Maximum blob area in pixels to keep. Default is 10000.
    Returns:
        A refined binary mask with only valid blobs retained.
    """
    # Find connected components in the binary mask
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    # Rebuild mask keeping only blobs within the area range (label 0 is background)
    refined = mask.copy()
    refined[:] = 0
    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if min_area <= area <= max_area:
            refined[labels == label] = 255

    return refined