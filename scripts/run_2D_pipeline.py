"""
Run the optimal detection + tracking pipeline end-to-end on one camera.
The Pipeline steps are:
    1. Open the camera's video, read frames.
    2. Two-pass YOLO with the fine-tuned model:
       - player pass at default resolution (imgsz=640) (class_ids = every non-ball class)
       - ball pass at imgsz=1280 with conf_threshold=0.1 (class_ids=[0])
    3. Merge the two passes into one TrackingOutput.
    4. Class-independent NMS (iou=0.5) to collapse duplicate identity-coded boxes.
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

import cv2
from ultralytics import YOLO

from src.detection.nms import class_independent_nms
from src.detection.yolo_detection import run_yolo_detection, yolo_to_detection_output
from src.tracking.deep_sort import apply_deep_sort
from src.tracking.label_resolution import resolve_track_labels
from src.types.tracking import merge_trackings
from src.utils.video import get_frames, open_video, produce_tracking_output_video


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the optimal tracking pipeline on one camera.")
    p.add_argument("--camera", required=True, help="Camera id, e.g. cam_13.")
    p.add_argument("--input-dir", default="data/videos",
                   help="Directory containing the source videos.")
    p.add_argument("--output-dir", default="results",
                   help="Root directory for produced videos.")
    p.add_argument("--model", default="models/fine_tuned_models/v2-yolo11m_finetuned.pt",
                   help="Path to the fine-tuned YOLO weights.")
    p.add_argument("--max-frames", type=int, default=-1,
                   help="-1 to read every frame, otherwise stop after N frames.")
    p.add_argument("--save-detection-video", action="store_true",
                   help="Also save the post-NMS detection video.")
    p.add_argument("--save-tracking-video", action="store_true",
                   help="Also save the tracking video before label resolution.")
    return p.parse_args()


def video_path_for(camera: str, input_dir: Path) -> Path:
    """Given a camera id like 'cam_13', return the expected path to its source video within `input_dir`."""
    num = camera.split("_", 1)[1]
    return input_dir / f"out{num}.mp4"


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    
    # Set up the output directory for this camera.
    out_dir = Path(args.output_dir) / "tracking" / args.camera
    out_dir.mkdir(parents=True, exist_ok=True)

    # Verify the input video exists.
    video_path = video_path_for(args.camera, input_dir)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    # 1. Open video, read frames, get fps.
    cap = open_video(str(video_path))
    max_frames = None if args.max_frames < 0 else args.max_frames
    frames_color, _ = get_frames(cap, max_frames=max_frames)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    print(f"Loaded {len(frames_color)} frames at {fps:.2f} fps from {video_path}")

    # 2. Two-pass YOLO.
    model = YOLO(args.model)
    player_classes = list(range(1, len(model.names)))  # everything except ball (class 0)

    # Run the player pass
    raw_player_results = run_yolo_detection(
        model, frames_color, 
        class_ids=player_classes
    )
    # Run the ball pass
    raw_ball_results = run_yolo_detection(
        model, frames_color,
        conf_threshold=0.1, inference_size=1280, class_ids=[0],
    )

    # 3. Convert YOLO outputs to DetectionOutput and merge them
    player_out = yolo_to_detection_output(
        raw_player_results, model,
        camera_id=args.camera, fps=fps, source="yolo_v11m_pt",
    )
    ball_out = yolo_to_detection_output(
        raw_ball_results, model,
        camera_id=args.camera, fps=fps, source="yolo_v11m_pt",
    )
    detection_output = merge_trackings(player_out, ball_out)

    # 4. Class-independent NMS.
    detection_output = class_independent_nms(detection_output, iou_threshold=0.5)

    # Optional: save the detection video with boxes drawn but before tracking.
    if args.save_detection_video:
        produce_tracking_output_video(frames_color, detection_output, str(out_dir / "detection.mp4"))

    # 5. DeepSORT.
    tracking_output = apply_deep_sort(
        detection_output,
        frames=frames_color,
        max_iou_distance=0.8,
        max_age=60,
        n_init=2,
    )

    if args.save_tracking_video:
        produce_tracking_output_video(frames_color, tracking_output, str(out_dir / "tracking.mp4"))

    # 6. Resolve labels.
    resolved_output = resolve_track_labels(tracking_output)

    # 7. Final tracking video — always.
    final_path = out_dir / "tracking_resolved.mp4"
    produce_tracking_output_video(frames_color, resolved_output, str(final_path))
    print(f"\nWrote final tracking video: {final_path}")


if __name__ == "__main__":
    main()
