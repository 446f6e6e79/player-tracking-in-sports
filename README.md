# Player Tracking in Sports - Multi-View Tracking and 3D Reconstruction

<div align="center">
    <strong>
        <a href="docs/report.pdf">View Full Report (PDF)</a>
        &nbsp;·&nbsp;
        <a href="https://huggingface.co/446f6e6e79/YOLO-basketball-fineTuned">🤗 Fine-tuned model on Hugging Face</a>
    </strong><br><br>
    <a href="docs/report.pdf">
        <img src="docs/media/report-preview.png" width="200" alt="Report preview">
    </a>
</div>

**Course:**  Computer Vision    
**Professors:**   Prof. Nicola Conci, Prof. Giulia Martinelli   
**Authors:** Andrea Blushi, Davide Donà 

---

## Overview

End-to-end pipeline for tracking basketball players and the ball across multiple synchronised camera views and reconstructing their 3D positions on the court. 

For the detection step, a fine-tuned YOLO is run as a two-pass scheme (player pass at 1280 px, ball-only pass at 1600 px), cleaned via class-independent NMS and merged. 

Tracks are produced with DeepSORT and stabilised by a cumulative-confidence label-resolution step.

The 3D reconstruction pipeline triangulates the merged 2D detections across views using the camera extrinsics, then smooths the resulting 3D tracks with a forward Kalman filter.

## Prerequisites

- **Python 3.11+** (matches `requirements.txt`: `numpy==2.4.3`, `torch==2.11.0`).
- **ffmpeg** on `PATH` for OpenCV video writing. Can be installed via `brew install ffmpeg` on macOS or downloaded from the [official site](https://ffmpeg.org/download.html) for Windows/Linux.
- **Match videos** in `data/videos/` as `out2.mp4`, `out4.mp4`, `out13.mp4` (cameras `cam_2`, `cam_4`, `cam_13`).
- **Fine-tuned weights** at `models/fine_tuned_models/best.pt` (default used by the pipeline script). Auto-downloaded from the [🤗 Hugging Face repo](https://huggingface.co/446f6e6e79/YOLO-basketball-fineTuned) on first run; you can also produce your own with the fine-tune flow below. 
- **Roboflow API key** (only needed if you run the evaluation cell in `notebook.ipynb`). Copy `.env.example` to `.env` and fill in `ANNOTATIONS_API_KEY`.

## Setup Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` is a full freeze of the dev environment, so this installs both runtime dependencies and the notebook tooling (Jupyter, pytest, mypy, …) in one shot.

## Running the Project

### Exploratory notebook — `notebook.ipynb`

```bash
jupyter lab notebook.ipynb
```

Walks through MOG2 baseline → YOLO baseline → fine-tuned two-pass detection → merge → tracking-evaluation → triangulation → evaluation.

Change `CURRENT_CAMERA_ID` in the second cell to switch between `cam_2`, `cam_4`, and `cam_13`.

### End-to-end 2D pipeline — `scripts/run_2D_pipeline.py`

```bash
python scripts/run_2D_pipeline.py --camera cam_13
```

Writes `results/<camera>/videos/tracking_resolved.mp4`. See the docstring at the top of the script for the full step list.

Useful flags:
- `--max-frames N` — stop after N frames (smoke testing).
- `--save-detection-video` / `--save-tracking-video` — also write the intermediate videos.
- `--model <path>` — override the default fine-tuned weights.

### End-to-end 3D reconstruction pipeline — `scripts/run_3D_reconstruction_pipeline.py`

```bash
python scripts/run_3D_reconstruction_pipeline.py --cameras cam_13 cam_4 cam_2
```
See the docstring at the top of the script for the full step list.

### Fine-tune YOLO

Two options.

**Colab (recommended — needs GPU).** Open [`finetune.ipynb`](finetune.ipynb) in Colab, set the runtime to GPU, add `ROBOFLOW_API_KEY` to Colab Secrets, and fill the `TODO_WORKSPACE` / `TODO_PROJECT` / `TODO_VERSION` placeholders. The final cell downloads `best.pt` to your machine — drop it into `models/fine_tuned_models/` to use it from the notebook or the pipeline script, or push it to the [🤗 Hugging Face repo](https://huggingface.co/446f6e6e79/YOLO-basketball-fineTuned) so other contributors can auto-download it on first run.

**Local.** Download a Roboflow YOLO export manually, then:

```bash
python scripts/finetune.py --data <export>/data.yaml --device 0
```

Defaults: `imgsz=1280`, `batch=4`, 300 epochs, `patience=30`. Copy the best checkpoint to `models/fine_tuned_models/best.pt` (or pass `--out` to the script).

## Repository Structure

```
computer-vision-project/
├── src/                          # Library code
│   ├── calibration/              # Camera params and extrinsics
│   ├── detection/                # MOG2, YOLO, NMS
│   ├── tracking/                 # DeepSORT, label resolution
│   ├── geometry/                 # Calibration / 3D helpers
│   ├── evaluation/               # Metrics against Roboflow annotations
│   ├── visualization/            # Rendering helpers
│   ├── types/                    # DetectionOutput / TrackingOutput dataclasses
│   └── utils/                    # Video I/O, visualization, annotations
├── scripts/
│   ├── run_2D_pipeline.py        # End-to-end 2D pipeline (CLI)
│   ├── run_3D_reconstruction_pipeline.py  # End-to-end 3D pipeline (CLI)
│   ├── calibrate_extrinsics.py   # Calibration helper
│   └── finetune.py               # YOLO fine-tuning (CLI)
├── notebook.ipynb                # Exploratory walkthrough
├── finetune.ipynb                # Colab orchestrator for scripts/finetune.py
├── models/                       # YOLO weights (git-ignored)
│   └── fine_tuned_models/         # Fine-tuned checkpoints (best.pt)
├── data/                         # Videos, calibration, annotations (git-ignored)
├── results/                      # Generated detection/tracking videos (git-ignored)
├── docs/                          # Report and LaTeX sources
│   ├── report.pdf                 # Full methodology
│   ├── LaTeX/                     # Report sources
│   ├── build/                     # LaTeX build outputs
│   └── media/                     # Figures and preview assets
├── requirements.txt
└── README.md
```
