import numpy as np

def iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute Intersections over Unions (IoU) between box a and each box in b.
    
    Parameters:
        a: array of shape (4,) representing a single box in (x1, y1, x2, y2) format.
        b: array of shape (N, 4) representing N boxes in (x1, y1, x2, y2) format.
    
    Returns:
        ious: array of shape (N,) with IoU between box a and each box in b.
    """
    # Get the coordinates of the intersection rectangle
    xx1 = np.maximum(a[0], b[:, 0]) 
    yy1 = np.maximum(a[1], b[:, 1])
    xx2 = np.minimum(a[2], b[:, 2])
    yy2 = np.minimum(a[3], b[:, 3])

    # Comput the area of intersection
    inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a + area_b - inter + 1e-9)
