"""
Run the optimal detection + tracking pipeline end-to-end on one camera.
The Pipeline steps are:
    1. Stream the camera's video in N-frame chunks (chunk_size, default 128).
    2. For each chunk run two YOLO passes with the fine-tuned model:
       - player pass at imgsz=1280 (class_ids = every non-ball class)
       - ball pass at imgsz=1600 with conf_threshold=0.2 (class_ids=[0])
       merge them, apply class-independent NMS (iou=0.75), then step
       DeepSORT one frame at a time so tracker state carries across chunks.
       Only lightweight detection/tracking metadata is retained — the chunk
       frames themselves are discarded as soon as the chunk is processed.
    3. Resolve per-track labels (cumulative-confidence vote) on the full
       TrackingOutput once streaming finishes.
    4. Render the final tracking video by re-streaming the input video and
       drawing the resolved boxes on each frame.

The intermediate detection / tracking videos are off by default — flag them on
explicitly when needed.
Usage example:
    python run_2D_pipeline.py --camera cam_13
"""
import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.cli import (
    add_force_arg,
    add_input_dir_arg,
    add_max_frames_arg,
    add_output_dir_arg,
    max_frames_or_none,
)
from src.detection.nms import class_independent_nms
from src.detection.yolo.detection import run_yolo_detection, yolo_to_detection_output
from src.detection.yolo.model import load_fine_tuned_yolo_model
from src.paths import CameraPaths, preflight_output_paths
from src.paths.model_paths import YOLO_FINE_TUNED_DIR
from src.tracking.deep_sort import build_deep_sort_tracker, step_deep_sort
from src.tracking.label_resolution import resolve_track_labels
from src.types.detection import DetectionOutput, merge_detections
from src.types.tracking import FrameTrackedDetections, TrackingOutput
from src.utils.logging import configure_logging, get_logger
from src.utils.video_io import stream_frame_chunks, video_fps
from src.visualization.drawing import draw_tracked_detections
from src.visualization.video_render import stream_tracking_output_video


logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the optimal tracking pipeline on one camera.")
    p.add_argument("--camera", required=True, help="Camera id, e.g. cam_13.")
    add_input_dir_arg(p)
    add_output_dir_arg(p)
    p.add_argument("--model", default="best.pt",
                   help="Path to the fine-tuned YOLO weights. If the file is missing, "
                        "best.pt is auto-downloaded from the Hugging Face repo.")
    p.add_argument("--yolo-batch", type=int, default=4,
                   help="YOLO inference batch size. 4 is laptop-safe; bump to 16-32 on a discrete GPU.")
    p.add_argument("--chunk-size", type=int, default=128,
                   help="Frames per processing chunk. Smaller = lower peak RAM.")
    add_max_frames_arg(p)
    p.add_argument("--save-detection-video", action="store_true",
                   help="Also save the post-NMS detection video.")
    p.add_argument("--save-tracking-video", action="store_true",
                   help="Also save the tracking video before label resolution.")
    add_force_arg(p)
    return p.parse_args()


def _detect_on_chunk(
    yolo_model,
    frames: list,
    *,
    player_classes: list[int],
    yolo_batch_size: int,
    camera: str,
    fps: float,
    model_name: str,
    frame_index_offset: int,
) -> DetectionOutput:
    """Run the two-pass YOLO + merge + NMS pipeline on a single chunk of frames."""
    logger.info("Running player pass...")
    raw_player = run_yolo_detection(
        yolo_model, frames, inference_size=1280,
        class_ids=player_classes, batch_size=yolo_batch_size,
    )
    logger.info("Running ball pass...")
    raw_ball = run_yolo_detection(
        yolo_model, frames,
        conf_threshold=0.2, inference_size=1600, class_ids=[0],
        batch_size=yolo_batch_size,
    )
    logger.info("Processing raw detections...")
    player_out = yolo_to_detection_output(
        raw_player, yolo_model,
        camera_id=camera, fps=fps, source=model_name,
        frame_index_offset=frame_index_offset,
    )
    ball_out = yolo_to_detection_output(
        raw_ball, yolo_model,
        camera_id=camera, fps=fps, source=model_name,
        frame_index_offset=frame_index_offset,
    )
    logger.info("Merging player and ball detections and applying NMS...")
    merged = merge_detections(player_out, ball_out)
    return class_independent_nms(merged, iou_threshold=0.75)


def _stream_annotated_frames(
    video_path: Path,
    output: DetectionOutput | TrackingOutput,
    chunk_size: int,
    max_frames: int | None,
):
    """Yield each input frame with the per-frame boxes drawn on top.

    Accepts either a DetectionOutput or a TrackingOutput — `draw_tracked_detections`
    handles both shapes (omitting `#track_id` when none is present).
    """
    by_index = {fd.frame_index: fd for fd in output.frames}
    for start, chunk in stream_frame_chunks(video_path, chunk_size=chunk_size, max_frames=max_frames):
        for offset, frame in enumerate(chunk):
            fd = by_index.get(start + offset)
            yield frame if fd is None else draw_tracked_detections(frame, fd)


def run_2d_pipeline(
    camera: str,
    *,
    input_dir: str | None = None,
    output_dir: str | None = None,
    model: str = "best.pt",
    max_frames: int | None = None,
    save_detection_video: bool = False,
    save_tracking_video: bool = False,
    force: bool = False,
    yolo_batch_size: int = 4,
    chunk_size: int = 128,
) -> None:
    """Programmatic entry point for the 2D detection+tracking pipeline (streaming)."""
    logger.info("Running 2D pipeline for camera %s with model %s...", camera, model)

    paths = CameraPaths.for_camera(
        camera,
        videos_input_dir=input_dir,
        results_dir=output_dir,
    )
    paths.mkdir()

    output_paths_to_check = [paths.tracking_video, paths.tracking_json]
    if save_detection_video:
        output_paths_to_check.append(paths.detection_video)
    if save_tracking_video:
        output_paths_to_check.append(paths.pre_resolution_video)

    # Check for existing outputs before doing any expensive compute, to avoid overwriting results
    preflight_output_paths(output_paths_to_check, force)

    if not paths.video.exists():
        raise FileNotFoundError(f"Video not found: {paths.video}")

    fps = video_fps(paths.video)
    logger.info("Source video %s reports %.2f fps", paths.video, fps)

    # Load the model once and discover the player class ids (everything but ball=0)
    model_path = YOLO_FINE_TUNED_DIR / model
    yolo_model = load_fine_tuned_yolo_model(model_path)
    player_classes = list(range(1, len(yolo_model.names)))

    tracker = build_deep_sort_tracker(
        max_iou_distance=0.7,
        max_age=30,
        n_init=2,
    )

    # Accumulators — these store *metadata only*; chunk frames are released
    # at the end of each chunk iteration.
    detection_frames: list = []
    tracking_frames: list[FrameTrackedDetections] = []

    total_frames = 0
    for start, chunk in stream_frame_chunks(paths.video, chunk_size=chunk_size,
                                            max_frames=max_frames):
        chunk_len = len(chunk)
        logger.info("Chunk @ frame %d (%d frames)", start, chunk_len)

        # Run detection on the chunk and accumulate the results
        chunk_dets = _detect_on_chunk(
            yolo_model, chunk,
            player_classes=player_classes,
            yolo_batch_size=yolo_batch_size,
            camera=camera, fps=fps, model_name=model,
            frame_index_offset=start,
        )
        detection_frames.extend(chunk_dets.frames)

        # Step DeepSORT one frame at a time so the tracker keeps state across chunks.
        for offset, frame in enumerate(chunk):
            fi = start + offset
            fd = chunk_dets.frames[offset]
            tracking_frames.append(step_deep_sort(tracker, fd, frame))
        total_frames += chunk_len
        del chunk

    if total_frames == 0:
        raise RuntimeError(f"No frames were read from {paths.video}.")
    logger.info("Processed %d frames across the streamed video.", total_frames)

    # Build the final outputs and write the videos
    detection_output = DetectionOutput(
        source=model, camera_id=camera, fps=fps, frames=detection_frames,
    )
    tracking_output = TrackingOutput(
        source=model, camera_id=camera, fps=fps, frames=tracking_frames,
    )

    if save_detection_video:
        logger.info("Streaming detection video...")
        stream_tracking_output_video(
            _stream_annotated_frames(paths.video, detection_output, chunk_size, max_frames),
            str(paths.detection_video), fps=fps,
        )

    if save_tracking_video:
        logger.info("Streaming pre-resolution tracking video...")
        stream_tracking_output_video(
            _stream_annotated_frames(paths.video, tracking_output, chunk_size, max_frames),
            str(paths.pre_resolution_video), fps=fps,
        )

    logger.info("Resolving track labels...")
    resolved_output = resolve_track_labels(tracking_output)

    logger.info("Writing resolved tracking output to %s...", paths.tracking_json)
    resolved_output.write(str(paths.tracking_json), overwrite=force)

    logger.info("Streaming final tracking video to %s...", paths.tracking_video)
    stream_tracking_output_video(
        _stream_annotated_frames(paths.video, resolved_output, chunk_size, max_frames),
        str(paths.tracking_video), fps=fps,
    )


def main() -> None:
    configure_logging()
    args = parse_args()
    run_2d_pipeline(
        camera=args.camera,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        model=args.model,
        max_frames=max_frames_or_none(args.max_frames),
        save_detection_video=args.save_detection_video,
        save_tracking_video=args.save_tracking_video,
        force=args.force,
        yolo_batch_size=args.yolo_batch,
        chunk_size=args.chunk_size,
    )


if __name__ == "__main__":
    main()
