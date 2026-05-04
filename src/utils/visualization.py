import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import display

from src.utils.drawing import draw_detections, draw_tracked_detections
from src.types.tracking import DetectionOutput, FrameDetections, FrameTrackedDetections, TrackingOutput
from src.types.evaluation import DetectionMetrics, EvaluationTrackingResult

def show_image(
    frame: cv2.Mat,
    title: str = ""
) -> None:
    """
    Displays an image using matplotlib. The image is expected to be in BGR format.
    Parameters:
        - frame: The image to display (in BGR format)
        - title: The title of the displayed image (default is an empty string)
    """
    # Convert the image from BGR to RGB format for displaying with matplotlib
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Display the image with the specified title and no axes
    plt.figure();plt.imshow(rgb);plt.title(title);plt.axis("off")
    plt.show()


def show_images(
    frames: list[cv2.Mat],
    frames_detections: DetectionOutput | TrackingOutput | list[FrameDetections] | list[FrameTrackedDetections] | None = None,
    *,
    titles: list[str] | None = None,
    n: int | None = None,
    indexes: list[int] | None = None,
) -> None:
    """Display a selection of images from a list of frames using matplotlib.

    Frames are expected in BGR format.

    Selection precedence:
        1. `indexes` — explicit list of frame indexes (used verbatim).
        2. `n`       — that many evenly-spaced frames (n=2 → first+last,
                       n=3 → first/middle/last, etc.).
        3. default   — first, middle, last (today's behavior).

    If `frames_detections` is provided, each selected frame is annotated before
    being shown. Accepts DetectionOutput or TrackingOutput directly, or the raw
    .frames list from either.

    Parameters:
        - frames: list of BGR images.
        - frames_detections: optional detections/tracks to overlay.
        - titles: optional list of titles, one per *selected* frame.
        - n: number of evenly-spaced frames to show.
        - indexes: explicit list of frame indexes to show.
    """
    if isinstance(frames_detections, (DetectionOutput, TrackingOutput)):
        frames_detections = frames_detections.frames

    if indexes is not None:
        selected = list(indexes)
    elif n is not None:
        selected = np.linspace(0, len(frames) - 1, n, dtype=int).tolist()
    else:
        selected = [0, len(frames) // 2, len(frames) - 1]

    fig, axes = plt.subplots(1, len(selected), figsize=(min(25, 5 * len(selected)), 5))
    # plt.subplots returns a bare Axes when squeezing 1×1; normalize to a list.
    if len(selected) == 1:
        axes = [axes]

    annotated = frames_detections is not None
    for i, frame_idx in enumerate(selected):
        frame = frames[frame_idx]
        if annotated:
            fd = frames_detections[frame_idx]
            frame = (draw_tracked_detections(frame, fd) if isinstance(fd, FrameTrackedDetections)
                     else draw_detections(frame, fd))
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        axes[i].imshow(rgb_frame)
        if titles is not None:
            axes[i].set_title(titles[i])
        elif annotated:
            axes[i].set_title(f"Frame {frame_idx} with Detections")
        else:
            axes[i].set_title(f"Frame {frame_idx}")
        axes[i].axis("off")
    plt.tight_layout()
    plt.show()


def show_hist(
    frame: cv2.Mat,
    title: str = ""
) -> None:
    """
    Displays the histogram of an image using matplotlib. The image is expected to be in BGR format.
        - If the image is grayscale, it shows a single histogram.
        - If the image is color, it shows separate histograms for each channel (blue, green, red).
    Parameters:
        - frame: The image to analyze (in BGR format)
        - title: The title of the displayed histogram (default is an empty string)
    """
    # Create a new figure and axis for plotting the histogram
    _, ax = plt.subplots()
    # Check if the image is grayscale (2D) or color (3D) and plot the appropriate histogram(s)
    if frame.ndim == 2:
        ax.plot(cv2.calcHist([frame], [0], None, [256], [0, 256]), color="gray")
    else:
        for i, color in enumerate(["blue", "green", "red"]):
            ax.plot(cv2.calcHist([frame], [i], None, [256], [0, 256]), color=color)

    ax.set_title(title)
    ax.set_xlim([0, 256])
    plt.show()


def show_detection_table(results: dict[str, EvaluationTrackingResult | DetectionMetrics]) -> None:
    """Display a detection metrics table with one row per camera/sequence.

    Parameters:
        - results: mapping of camera name to EvaluationTrackingResult or DetectionMetrics.
    """
    rows = []
    for camera, result in results.items():
        d = result.detection if isinstance(result, EvaluationTrackingResult) else result
        rows.append({
            "Camera": camera,
            "TP": d.tp, "FP": d.fp, "FN": d.fn,
            "Precision": round(d.precision, 3), "Recall": round(d.recall, 3),
            "F1": round(d.f1, 3), "Mean IoU": round(d.mean_iou, 3),
        })
    display(pd.DataFrame(rows).set_index("Camera"))


def show_identity_table(results: dict[str, EvaluationTrackingResult]) -> None:
    """Display an identity (IDF1) metrics table with one row per camera/sequence.

    Parameters:
        - results: mapping of camera name to EvaluationTrackingResult.
    """
    rows = []
    for camera, result in results.items():
        id_ = result.identity
        rows.append({
            "Camera": camera,
            "TP": id_.tp, "FP": id_.fp, "FN": id_.fn,
            "IDP": round(id_.precision, 3), "IDR": round(id_.recall, 3), "IDF1": round(id_.f1, 3),
        })
    display(pd.DataFrame(rows).set_index("Camera"))


def show_hota_table(results: dict[str, EvaluationTrackingResult]) -> None:
    """Display a HOTA metrics table with one row per camera/sequence.

    Per-alpha breakdowns are omitted; use HOTAMetrics.hota_per_alpha etc. for those.

    Parameters:
        - results: mapping of camera name to EvaluationTrackingResult.
    """
    rows = []
    for camera, result in results.items():
        h = result.hota
        rows.append({
            "Camera": camera,
            "HOTA": round(h.hota, 3), "DetA": round(h.deta, 3),
            "AssA": round(h.assa, 3), "LocA": round(h.loca, 3),
        })
    display(pd.DataFrame(rows).set_index("Camera"))
