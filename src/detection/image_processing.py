import cv2


def make_clahe(
    clip_limit: float,
    tile_grid_size: tuple[int, int],
) -> cv2.CLAHE:
    """Build a CLAHE object with the given parameters."""
    return cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)


def make_morph_kernel(size: int) -> cv2.Mat:
    """Build an elliptical structuring element of `size × size`."""
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def normalize_illumination(frame: cv2.Mat, *, clahe: cv2.CLAHE) -> cv2.Mat:
    """Apply CLAHE-based illumination normalization to a BGR frame."""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


def opening_closing(
        mask: cv2.Mat,
        opening_kernel_size: int,
        closing_kernel_size: int,
        *,
        opening_kernel: cv2.Mat | None = None,
        closing_kernel: cv2.Mat | None = None,
    ) -> cv2.Mat:
    """
    Single-frame variant of opening_closing. Takes precomputed structuring elements so the
    same kernels can be reused across many frames in a fused pipeline.
    Parameters:
        - mask: A single binary mask.
        - opening_kernel_size: Kernel size for morphological opening.
        - closing_kernel_size: Kernel size for morphological closing.
        - opening_kernel / closing_kernel: optional precomputed kernels; built
          from the size args when omitted.
    Returns:
        A binary mask with opening then closing applied.
    """
    if opening_kernel is None:
        opening_kernel = make_morph_kernel(opening_kernel_size)
    if closing_kernel is None:
        closing_kernel = make_morph_kernel(closing_kernel_size)

    # Apply morphological opening to remove small noise, then closing to fill small holes
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, opening_kernel)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, closing_kernel)
    return closed


def refine_blobs(
    mask: cv2.Mat,
    min_area: int,
    max_area: int,
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
