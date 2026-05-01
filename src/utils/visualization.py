import cv2
import matplotlib.pyplot as plt

from src.types.tracking import Frame_Detections

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
    titles: list[str] | None = None,
    is_yolo: bool = False
) -> None:
    """
    Displays first, middle, and last images from a list of frames using matplotlib. 
    The images are expected to be in BGR format.
    Parameters:
        - frames: A list of images to display (in BGR format)
        - titles: An optional list of titles for each image (default is None, which means no titles)
        - is_yolo: A boolean indicating whether the images are YOLO annotated frames (default is False)
    """
    selected_frames_idxs = [0, len(frames)//2, len(frames)-1]
    _, axes = plt.subplots(1, 3, figsize=(20, 5))

    for i, frame_idx in enumerate(selected_frames_idxs):
        frame = frames[frame_idx]
        
        if is_yolo:
            # If the frame is a YOLO annotated frame, we need to call the plot() method to get the visualized image
            frame = frame.plot()
        # Convert the image from BGR to RGB format for displaying with matplotlib
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        axes[i].imshow(rgb_frame)
        if titles is not None:
            axes[i].set_title(titles[i])
        # Set default title to indicate frame index if titles are not provided
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


def draw_detections(
    frame: cv2.Mat,
    frame_detections: Frame_Detections,
    draw_conf: bool = True,
) -> cv2.Mat:
    """Given a frame and its corresponding Frame_Detections, draw the bounding boxes and confidence scores on the frame.
    Parameters:
        - frame: BGR image to annotate (not modified in place)
        - frame_detections: Frame_Detections containing detections to draw
        - draw_conf: whether to overlay confidence scores
    Returns:
        Annotated copy of the input frame.
    """
    # Start with a copy of the original frame to draw on
    annotated = frame.copy()
    bbox_color = (0, 255, 0)  # green for all pure detections

    # For each detection in the frame, draw its bounding box and optionally its confidence score
    for detection in frame_detections.detections:
        x1, y1, x2, y2 = map(int, detection.get_bbox_tuple())
        cv2.rectangle(annotated, (x1, y1), (x2, y2), bbox_color, 2)

        # If draw_conf is True, overlay the confidence score as text near the bounding box
        if draw_conf:
            label = f"{detection.confidence:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated, (x1, y1 - th - 6), (x1 + tw + 4, y1), bbox_color, -1)
            cv2.putText(annotated, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    return annotated
