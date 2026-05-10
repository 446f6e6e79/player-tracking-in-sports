"""
Run the optimal detection + tracking pipeline end-to-end on one camera.
The Pipeline steps are:
    1. Open the camera's video, read frames.
    2. Two-pass YOLO with the fine-tuned model:
       - player pass at default resolution (imgsz=1280) (class_ids = every non-ball class)
       - ball pass at imgsz=1600 with conf_threshold=0.2 (class_ids=[0])
    3. Merge the two passes into one DetectionOutput.
    4. Class-independent NMS (iou=0.75) to collapse duplicate identity-coded boxes.
    5. DeepSORT (max_iou_distance=0.8, max_age=60, n_init=2).
    6. Resolve per-track labels (cumulative-confidence vote).
    7. Render the final tracking video. 

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

import cv2

from src.detection.nms import class_independent_nms
from src.detection.yolo.detection import run_yolo_detection, yolo_to_detection_output
from src.detection.yolo.model import load_fine_tuned_yolo_model
from src.tracking.deep_sort import apply_deep_sort
from src.tracking.label_resolution import resolve_track_labels
from src.types.detection import merge_detections
from src.utils.video_io import get_frames, open_video
from src.visualization.video_render import produce_tracking_output_video

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the optimal tracking pipeline on one camera.")
    p.add_argument("--camera", required=True, help="Camera id, e.g. cam_13.")
    p.add_argument("--input-dir", default="data/videos",
                   help="Directory containing the source videos.")
    p.add_argument("--output-dir", default="results",
                   help="Root directory for produced videos.")
    p.add_argument("--model", default="best.pt",
                   help="Path to the fine-tuned YOLO weights. If the file is missing, "
                        "best.pt is auto-downloaded from the Hugging Face repo.")
    p.add_argument("--max-frames", type=int, default=-1,
                   help="-1 to read every frame, otherwise stop after N frames.")
    p.add_argument("--save-detection-video", action="store_true",
                   help="Also save the post-NMS detection video.")
    p.add_argument("--save-tracking-video", action="store_true",
                   help="Also save the tracking video before label resolution.")
    p.add_argument("--force", action="store_true",
                   help="Overwrite existing pipeline outputs instead of failing.")
    return p.parse_args()

def _preflight_output_paths(paths: list[Path], force: bool) -> None:
    """Validate all output paths up front so we fail before expensive compute."""
    for path in paths:
        if path.exists() and not force:
            raise FileExistsError(
                f"Refusing to overwrite existing file: {path}. Pass --force to replace it."
            )


def video_path_for(camera: str, input_dir: Path) -> Path:
    """Given a camera id like 'cam_13', return the expected path to its source video within `input_dir`."""
    num = camera.split("_", 1)[1]
    return input_dir / f"out{num}.mp4"


def main() -> None:
    args = parse_args()
    print(f"Running 2D pipeline for camera {args.camera} with model {args.model}...")

    input_dir = Path(args.input_dir)
    
    # Set up the output directory for this camera
    video_out_dir = Path(args.output_dir) / "tracking" / args.camera
    video_out_dir.mkdir(parents=True, exist_ok=True)
    serialized_track_out_dir = video_out_dir / "serialized" / args.camera
    serialized_track_out_dir.mkdir(parents=True, exist_ok=True)

    # Build paths to script outputs
    final_video_path = video_out_dir / "tracking_resolved.mp4"
    final_json_path = serialized_track_out_dir / "tracking.json"
    output_paths_to_check = [final_video_path, final_json_path]
    
    # Optional intermediate videos
    if args.save_detection_video:
        detection_video_path = video_out_dir / "detection.mp4"
        output_paths_to_check.append(detection_video_path)
    
    if args.save_tracking_video:
        tracking_video_path = video_out_dir / "tracking_pre_resolution.mp4"
        output_paths_to_check.append(tracking_video_path)

    # Check for existing outputs before doing any expensive compute, to avoid overwriting results
    _preflight_output_paths(output_paths_to_check, args.force)

    # Verify the input video exists
    video_path = video_path_for(args.camera, input_dir)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    # 1. Open video, read frames, get fps
    cap = open_video(str(video_path))
    max_frames = None if args.max_frames < 0 else args.max_frames
    frames_color, _ = get_frames(cap, max_frames=max_frames)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    print(f"Loaded {len(frames_color)} frames at {fps:.2f} fps from {video_path}")

    # 2. Two-pass YOLO
    MODEL_PATH = Path(f"models/fine_tuned_models/{args.model}")
    model = load_fine_tuned_yolo_model(MODEL_PATH)
    player_classes = list(range(1, len(model.names)))  # everything except ball (class 0)

    # Run the player pass
    print("Running player detection pass...")
    raw_player_results = run_yolo_detection(
        model, frames_color, inference_size=1280,
        class_ids=player_classes
    )
    print("Player detection pass completed.")
    
    print("Running ball detection pass...")
    # Run the ball pass
    raw_ball_results = run_yolo_detection(
        model, frames_color,
        conf_threshold=0.2, inference_size=1600, class_ids=[0],
    )
    print("Ball detection pass completed.")

    # 3. Convert YOLO outputs to DetectionOutput and merge them
    print("Merging player and ball detections...")
    player_out = yolo_to_detection_output(
        raw_player_results, model,
        camera_id=args.camera, fps=fps, source=args.model,
    )
    ball_out = yolo_to_detection_output(
        raw_ball_results, model,
        camera_id=args.camera, fps=fps, source=args.model,
    )
    detection_output = merge_detections(player_out, ball_out)

    # 4. Class-independent NMS
    print("Applying class-independent NMS to merge duplicate boxes...")
    detection_output = class_independent_nms(detection_output, iou_threshold=0.75)

    # Optional: save the detection video with boxes drawn but before tracking
    if args.save_detection_video:
        print("Saving detection video...")
        produce_tracking_output_video(frames_color, detection_output, str(detection_video_path))

    # 5. DeepSORT
    print("Applying DeepSORT...")
    tracking_output = apply_deep_sort(
        detection_output,
        frames=frames_color,
        max_iou_distance=0.8,
        max_age=60,
        n_init=2,
    )
    print("DeepSORT tracking completed.")

    if args.save_tracking_video:
        print("Saving tracking video before label resolution...")
        produce_tracking_output_video(frames_color, tracking_output, str(tracking_video_path))

    # 6. Resolve labels
    print("Resolving track labels...")
    resolved_output = resolve_track_labels(tracking_output)

    # 7. Persist the resolved tracking output (JSON) and render the final video
    print(f"Writing resolved tracking output to {final_json_path}...")
    resolved_output.write(str(final_json_path), overwrite=args.force)

    print(f"Saving final tracking video with resolved labels to {final_video_path}...")
    produce_tracking_output_video(frames_color, resolved_output, str(final_video_path))

if __name__ == "__main__":
    main()
