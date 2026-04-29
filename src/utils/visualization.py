import cv2
import matplotlib.pyplot as plt

def show_image(frame, title=""):
    """
    Displays an image
    Parameters:
        - frame: The image to display (in BGR format)
        - title: The title of the displayed image (default is an empty string)
    """
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    plt.figure()
    plt.imshow(rgb)
    plt.title(title)
    plt.axis("off")
    plt.show()


def show_hist(frame, title=""):
    """
    Displays the histogram of an image. 
    If the image is grayscale, it shows a single histogram. If the image is color, it shows separate histograms for each channel (blue, green, red).
    Parameters:
        - frame: The image to analyze (in BGR format)
        - title: The title of the displayed histogram (default is an empty string)
    """
    fig, ax = plt.subplots()
    if frame.ndim == 2:
        ax.plot(cv2.calcHist([frame], [0], None, [256], [0, 256]), color="gray")
    else:
        for i, color in enumerate(["blue", "green", "red"]):
            ax.plot(cv2.calcHist([frame], [i], None, [256], [0, 256]), color=color)
    ax.set_title(title)
    ax.set_xlim([0, 256])
    plt.show()
