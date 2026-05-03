import os
import cv2

from src.types.tracking import DetectionOutput, TrackingOutput
from src.utils.drawing import draw_detections, draw_tracked_detections

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


def save_video(
        frames: list[cv2.Mat],
        output_path: str,
        fps: int = 25
    ) -> None:
    """
    Saves a list of frames as a video file.
    Parameters:
        - frames (list[cv2.Mat]): A list of frames to save as a video
        - output_path (str): The path where the video will be saved.
        - fps (int): The frames per second for the output video. Default is 25 (Given by the video source).
    """

    # Get the dimensions of the frames
    height, width = frames[0].shape[:2]

    # Ensure the output directory exists, otherwise create it
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Create a VideoWriter object to write the video file
    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    # Initialize the VideoWriter object with the specified output path, codec, fps, and frame size
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height), True)
    if not out.isOpened():
        raise RuntimeError(f"Could not open VideoWriter for: {output_path}")

    # Write each frame to the video file
    for frame in frames:
        out.write(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR) if frame.ndim == 2 else frame)
    print(f"Video saved successfully at: {output_path}")

    # Release the VideoWriter object to finalize the video file
    out.release()


def produce_detection_output_video(
    frames: list[cv2.Mat],
    detection_output: DetectionOutput,
    output_path: str,
    fps: float | None = None,
    draw_conf: bool = True,
) -> None:
    """Produce an annotated output video from frames and a pre-tracking detection output.
    Parameters:
        - frames: list of original BGR frames (must match len(detection_output.frames))
        - detection_output: per-frame detections (no track_ids)
        - output_path: path for the output MP4
        - fps: frame rate; falls back to detection_output.fps if None
        - draw_conf: whether to overlay confidence scores on boxes
    """
    out_fps = fps if fps is not None else detection_output.fps
    annotated = [
        draw_detections(frame, frame_detections, draw_conf)
        for frame, frame_detections in zip(frames, detection_output.frames)
    ]
    save_video(annotated, output_path, int(out_fps))


def produce_tracking_output_video(
    frames: list[cv2.Mat],
    output: DetectionOutput | TrackingOutput,
    output_path: str,
    fps: float | None = None,
) -> None:
    """Produce an annotated output video from frames and either a detection or
    tracking output — the detection variant simply omits the '#track_id' caption.

    Bounding boxes are colored by team (Ball=yellow, Red=red, White=white,
    Referee=orange). Each box is captioned just above its top edge with
    '{jersey_number} #{track_id} {confidence:.2f}'. Class labels are expected
    to follow the fine-tuned model's schema ('Red_11', 'White_2', 'Refree_1',
    'Ball'); unknown labels fall back to a gray box with the raw class name.

    Parameters:
        - frames: list of original BGR frames (must match len(output.frames))
        - output: per-frame detections — DetectionOutput or TrackingOutput
        - output_path: path for the output MP4
        - fps: frame rate; falls back to output.fps if None
    """
    out_fps = fps if fps is not None else output.fps
    annotated = [
        draw_tracked_detections(frame, frame_detections)
        for frame, frame_detections in zip(frames, output.frames)
    ]
    save_video(annotated, output_path, int(out_fps))