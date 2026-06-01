"""
Run the optimal detection + tracking pipeline end-to-end on one camera.
The Pipeline steps are:
    1. Stream the camera's video in N-frame chunks (chunk_size from config).
    2. For each chunk run two YOLO passes with the fine-tuned model:
       - player pass (class_ids = every non-ball class)
       - ball pass with lower conf_threshold
       merge them, apply class-independent NMS, then step
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
    python scripts/run_2D_pipeline.py --camera cam_13
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
from src.config import get_config
from src.detection.nms import class_independent_nms
from src.detection.yolo.detection import run_yolo_detection, yolo_to_detection_output
from src.detection.yolo.model import load_fine_tuned_yolo_model
from src.paths import CameraPaths, preflight_output_paths
from src.paths.model_paths import YOLO_FINE_TUNED_DIR
from src.tracking.deep_sort import build_deep_sort_tracker, step_deep_sort
from src.tracking.label_resolution import resolve_track_labels
from src.tracking.trajectory_smoothing import smooth_tracking_output
from src.types.detection import DetectionOutput, FrameDetections, merge_detections
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
    p.add_argument("--model", default=None,
                   help="Path to the fine-tuned YOLO weights (default: from config). "
                        "If the file is missing, it is auto-downloaded from the Hugging Face repo.")
    p.add_argument("--yolo-batch", type=int, default=None,
                   help="YOLO inference batch size (default: from config).")
    p.add_argument("--chunk-size", type=int, default=None,
                   help="Frames per processing chunk (default: from config). Smaller = lower peak RAM.")
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
) -> tuple[DetectionOutput, list[list]]:
    """Run the two-pass YOLO + merge + NMS pipeline on a single chunk of frames.

    Returns:
        - main DetectionOutput (high-confidence detections, NMS-merged with ball)
        - per-frame list of low-confidence player detections for ByteTrack second pass
    """
    cfg = get_config()
    tp = cfg.detection.two_pass
    high_threshold = cfg.detection.yolo.conf_threshold

    # Player pass: run at the lower ByteTrack threshold so we get both sets in one
    # YOLO call, then split by confidence in Python.
    logger.info("Running player pass...")
    raw_player = run_yolo_detection(
        yolo_model, frames,
        conf_threshold=tp.player_low_conf_threshold,
        inference_size=tp.player_inference_size,
        iou_threshold=cfg.detection.yolo.iou_threshold,
        batch_size=yolo_batch_size,
        class_ids=player_classes,
    )
    logger.info("Running ball pass...")
    raw_ball = run_yolo_detection(
        yolo_model, frames,
        conf_threshold=tp.ball_conf_threshold,
        inference_size=tp.ball_inference_size,
        iou_threshold=cfg.detection.yolo.iou_threshold,
        batch_size=yolo_batch_size,
        class_ids=[cfg.classes.ball_class_id],
    )
    logger.info("Processing raw detections...")
    all_player_out = yolo_to_detection_output(
        raw_player, yolo_model,
        camera_id=camera, fps=fps, source=model_name,
        frame_index_offset=frame_index_offset,
    )
    ball_out = yolo_to_detection_output(
        raw_ball, yolo_model,
        camera_id=camera, fps=fps, source=model_name,
        frame_index_offset=frame_index_offset,
    )

    # Split player detections into high-conf (main pipeline) and low-conf (ByteTrack).
    high_conf_frames = []
    low_conf_per_frame = []
    for fd in all_player_out.frames:
        high_dets = [d for d in fd.detections if d.confidence >= high_threshold]
        low_dets = [d for d in fd.detections if d.confidence < high_threshold]
        high_conf_frames.append(FrameDetections(frame_index=fd.frame_index, detections=high_dets))
        low_conf_per_frame.append(low_dets)

    player_out = DetectionOutput(
        source=all_player_out.source, camera_id=all_player_out.camera_id,
        fps=all_player_out.fps, frames=high_conf_frames,
    )

    logger.info("Merging player and ball detections and applying NMS...")
    merged = merge_detections(player_out, ball_out)
    return class_independent_nms(merged, iou_threshold=tp.merge_nms_iou), low_conf_per_frame


def _stream_annotated_frames(
    video_path: Path,
    output: DetectionOutput | TrackingOutput,
    chunk_size: int,
    max_frames: int | None,
):
    """Yield each input frame with the per-frame boxes drawn on top."""
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
    model: str | None = None,
    max_frames: int | None = None,
    save_detection_video: bool = False,
    save_tracking_video: bool = False,
    force: bool = False,
    yolo_batch_size: int | None = None,
    chunk_size: int | None = None,
) -> None:
    """Programmatic entry point for the 2D detection+tracking pipeline (streaming)."""
    cfg = get_config()

    # CLI overrides take precedence; fall back to config values.
    resolved_model = model or cfg.models.yolo_filename
    resolved_batch = yolo_batch_size if yolo_batch_size is not None else cfg.pipeline.yolo_batch_size
    resolved_chunk = chunk_size if chunk_size is not None else cfg.pipeline.chunk_size

    logger.info("Running 2D pipeline for camera %s with model %s...", camera, resolved_model)

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

    preflight_output_paths(output_paths_to_check, force)

    if not paths.video.exists():
        raise FileNotFoundError(f"Video not found: {paths.video}")

    fps = video_fps(paths.video)
    logger.info("Source video %s reports %.2f fps", paths.video, fps)

    model_path = YOLO_FINE_TUNED_DIR / resolved_model
    yolo_model = load_fine_tuned_yolo_model(model_path)
    player_classes = list(range(1, len(yolo_model.names)))

    ds = cfg.tracking.deep_sort
    tracker = build_deep_sort_tracker(
        max_iou_distance=ds.max_iou_distance,
        max_appearance_distance=ds.max_appearance_distance,
        max_age=ds.max_age,
        n_init=ds.n_init,
        feature_budget=ds.feature_budget,
    )

    detection_frames: list = []
    tracking_frames: list[FrameTrackedDetections] = []

    total_frames = 0
    for start, chunk in stream_frame_chunks(paths.video, chunk_size=resolved_chunk,
                                            max_frames=max_frames):
        chunk_len = len(chunk)
        logger.info("Chunk @ frame %d (%d frames)", start, chunk_len)

        chunk_dets, low_conf_by_frame = _detect_on_chunk(
            yolo_model, chunk,
            player_classes=player_classes,
            yolo_batch_size=resolved_batch,
            camera=camera, fps=fps, model_name=resolved_model,
            frame_index_offset=start,
        )
        detection_frames.extend(chunk_dets.frames)

        for offset, frame in enumerate(chunk):
            fd = chunk_dets.frames[offset]
            tracking_frames.append(
                step_deep_sort(tracker, fd, frame, low_conf_detections=low_conf_by_frame[offset])
            )
        total_frames += chunk_len
        del chunk

    if total_frames == 0:
        raise RuntimeError(f"No frames were read from {paths.video}.")
    logger.info("Processed %d frames across the streamed video.", total_frames)

    detection_output = DetectionOutput(
        source=resolved_model, camera_id=camera, fps=fps, frames=detection_frames,
    )
    tracking_output = TrackingOutput(
        source=resolved_model, camera_id=camera, fps=fps, frames=tracking_frames,
    )

    if save_detection_video:
        logger.info("Streaming detection video...")
        stream_tracking_output_video(
            _stream_annotated_frames(paths.video, detection_output, resolved_chunk, max_frames),
            str(paths.detection_video), fps=fps,
        )

    if save_tracking_video:
        logger.info("Streaming pre-resolution tracking video...")
        stream_tracking_output_video(
            _stream_annotated_frames(paths.video, tracking_output, resolved_chunk, max_frames),
            str(paths.pre_resolution_video), fps=fps,
        )

    ts = cfg.tracking.trajectory_smoothing
    logger.info("Smoothing 2D trajectories (gap fill + EMA)...")
    tracking_output = smooth_tracking_output(
        tracking_output,
        max_gap=ts.max_gap,
        ema_alpha=ts.ema_alpha,
    )

    logger.info("Resolving track labels...")
    resolved_output = resolve_track_labels(tracking_output)

    logger.info("Writing resolved tracking output to %s...", paths.tracking_json)
    resolved_output.write(str(paths.tracking_json), overwrite=force)

    logger.info("Streaming final tracking video to %s...", paths.tracking_video)
    stream_tracking_output_video(
        _stream_annotated_frames(paths.video, resolved_output, resolved_chunk, max_frames),
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
