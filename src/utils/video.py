import os
import cv2

def open_video(video_path: str) -> cv2.VideoCapture:
    """
    Given a video file path, opens the video and returns a VideoCapture object.
    Parameters:
        - video_path (str): The path to the video file.
    Returns:
        - cap (cv2.VideoCapture): The VideoCapture object for the video.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    return cap

def get_frames(
    cap: cv2.VideoCapture, 
    max_frames: int = None, 
    to_gray: bool = False
) -> tuple[list, list]:
    """
    Given a VideoCapture object, get the frames from the video and return them as lists of color and grayscale frames.
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
    
    # Read frames from the video until the end or until max_frames is reached
    while max_frames is None or count < max_frames:
        ret, frame = cap.read()
        # If ret is False, it means we have reached the end of the video or there was an error reading a frame
        if not ret:
            break
        frames_color.append(frame)
        # If to_gray is True, convert the frame to grayscale and add it to the frames_gray list
        if to_gray:
            frames_gray.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        count += 1
    return frames_color, frames_gray
