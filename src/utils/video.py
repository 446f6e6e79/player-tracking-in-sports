import os
import cv2

def open_video(video_path):
    """
    Opens a video file and returns the VideoCapture object.
    Parameters:
        - video_path (str): The path to the video file.
    Returns:
        - cap (cv2.VideoCapture): The VideoCapture object for the video.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    return cap


def read_frames(cap, max_frames=None, to_gray=False):
    """
    Reads frames from a VideoCapture object.
    Parameters:
        - cap (cv2.VideoCapture): The VideoCapture object to read from.
        - max_frames (int, optional): The maximum number of frames to read. If None, reads all frames.
        - to_gray (bool, optional): Whether to convert frames to grayscale. Default is False.
    Returns:
        - frames_color (list): A list of color frames read from the video.
        - frames_gray (list): A list of grayscale frames read from the video (empty if to_gray is False).
    """
    frames_color = []
    frames_gray = []
    count = 0
    while max_frames is None or count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frames_color.append(frame)
        if to_gray:
            frames_gray.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        count += 1
    return frames_color, frames_gray
