import os
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np


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
    max_frames: int | None = None,
) -> list[np.ndarray]:
    """Read color frames from `cap` and return them as a list of BGR ndarrays.

    Parameters:
        - cap: the VideoCapture object to read from.
        - max_frames: stop after this many frames; None reads to end of stream.
    """
    frames: list[np.ndarray] = []
    count = 0
    while max_frames is None or count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        count += 1
    return frames


def load_video_frames(
    video_path: str | Path,
    max_frames: int | None = None,
) -> tuple[list[np.ndarray], float]:
    """Open a video, decode all (or `max_frames`) frames, return `(frames, fps)`.

    Wraps the common `open_video` + `get_frames` + `cap.get(FPS)` + release dance
    that both pipeline scripts duplicate.
    """
    cap = open_video(str(video_path))
    try:
        frames = get_frames(cap, max_frames=max_frames)
        fps = cap.get(cv2.CAP_PROP_FPS)
    finally:
        cap.release()
    return frames, fps


def video_fps(video_path: str | Path) -> float:
    """Return the frame rate reported by the video file at `video_path`."""
    cap = open_video(str(video_path))
    try:
        return cap.get(cv2.CAP_PROP_FPS)
    finally:
        cap.release()


def stream_frame_chunks(
    video_path: str | Path,
    chunk_size: int,
    max_frames: int | None = None,
) -> Iterator[tuple[int, list[np.ndarray]]]:
    """Stream a video as chunks of frames without ever holding the full video in RAM.

    Yields `(start_index, frames_in_chunk)` tuples where `start_index` is the
    absolute (0-based) frame index of the chunk's first frame. The final chunk
    may be shorter than `chunk_size`. Honors `max_frames` if provided.
    """
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")
    cap = open_video(str(video_path))
    try:
        produced = 0
        start_index = 0
        chunk: list[np.ndarray] = []
        while True:
            if max_frames is not None and produced >= max_frames:
                break
            ret, frame = cap.read()
            if not ret:
                break
            chunk.append(frame)
            produced += 1
            if len(chunk) >= chunk_size:
                yield start_index, chunk
                start_index += len(chunk)
                chunk = []
        if chunk:
            yield start_index, chunk
    finally:
        cap.release()


def load_first_frame(video_path: str | Path) -> np.ndarray:
    """Read and return the first frame of a video as a BGR ndarray.

    Raises RuntimeError if the video opens but yields no frames.
    """
    cap = open_video(str(video_path))
    try:
        frames = get_frames(cap, max_frames=1)
    finally:
        cap.release()
    if not frames:
        raise RuntimeError(f"Could not read any frame from {video_path}")
    return frames[0]


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
