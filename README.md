# Player Tracking in Sports - Multi-View Tracking and 3D Reconstruction

<div align="center">
    <strong>
        <a href="docs/report/report.pdf">View Full Report (PDF)</a>
    </strong><br><br>
    <a href="docs/report/report.pdf">
        <img src="docs/media/report-preview.png" width="200" alt="Report preview">
    </a>
</div>

**Course:**  Computer Vision    
**Professors:**   Prof. Nicola Conci, Prof. Giulia Martinelli   
**Authors:** Andrea Blushi, Davide Donà 

---

## Overview

End-to-end pipeline for tracking basketball players and the ball across multiple synchronised camera views and reconstructing their 3D positions on the court. 

For the detection step, a fine-tuned YOLOv11m is run as a two-pass scheme (player pass at 640 px, ball-only pass at 1280 px), merged via class-independent NMS. 

Tracks are produced with DeepSORT and stabilised by a cumulative-confidence label-resolution step. 


## Prerequisites

- **Python 3.11+** (matches `requirements.txt`: `numpy==2.4.3`, `torch==2.11.0`).
- **ffmpeg** on `PATH` for OpenCV video writing.
- **Match videos** in `data/videos/` as `out2.mp4`, `out4.mp4`, `out13.mp4` (cameras `cam_2`, `cam_4`, `cam_13`).
- **Fine-tuned weights** at `models/fine_tuned_models/v2-yolo11m_finetuned.pt` (default used by the pipeline script). Once can be produced with the fine-tune flow below. `models/` is git-ignored.
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

Walks through MOG2 baseline → YOLOv11 baseline → fine-tuned two-pass detection → merge → evaluation. Change `CURRENT_CAMERA_ID` in the second cell to switch between `cam_2`, `cam_4`, and `cam_13`.

### End-to-end 2D pipeline — `scripts/run_2D_pipeline.py`

```bash
python scripts/run_2D_pipeline.py --camera cam_13
```

Writes `results/tracking/<camera>/tracking_resolved.mp4`. See the docstring at the top of the script for the full step list.

Useful flags:
- `--max-frames N` — stop after N frames (smoke testing).
- `--save-detection-video` / `--save-tracking-video` — also write the intermediate videos.
- `--model <path>` — override the default fine-tuned weights.

### Fine-tune YOLOv11m

Two options.

**Colab (recommended — needs GPU).** Open [`finetune.ipynb`](finetune.ipynb) in Colab, set the runtime to GPU, add `ROBOFLOW_API_KEY` to Colab Secrets, and fill the `TODO_WORKSPACE` / `TODO_PROJECT` / `TODO_VERSION` placeholders. The final cell downloads `yolo11m_finetuned.pt` to your machine — drop it into `models/fine_tuned_models/` to use it from the notebook or the pipeline script.

**Local.** Download a Roboflow YOLO export manually, then:

```bash
python scripts/finetune.py --data <export>/data.yaml --device 0
```

Defaults: `imgsz=1280`, `batch=4`, 300 epochs, `patience=30`. The best checkpoint is copied to `models/yolo11m_finetuned.pt`.

## Repository Structure

```
player-tracking-in-sports/
├── src/                          # Library code
│   ├── detection/                # MOG2, YOLO, NMS
│   ├── tracking/                 # DeepSORT, label resolution
│   ├── geometry/                 # Calibration / 3D helpers
│   ├── evaluation/               # Metrics against Roboflow annotations
│   ├── types/                    # DetectionOutput / TrackingOutput dataclasses
│   └── utils/                    # Video I/O, visualization, annotations
├── scripts/
│   ├── run_2D_pipeline.py        # End-to-end 2D pipeline (CLI)
│   └── finetune.py               # YOLOv11m fine-tuning (CLI)
├── notebook.ipynb                # Exploratory walkthrough
├── finetune.ipynb                # Colab orchestrator for scripts/finetune.py
├── models/                       # YOLO weights (git-ignored)
├── data/                         # Videos, calibration, annotations (git-ignored)
├── results/                      # Generated detection/tracking videos (git-ignored)
├── docs/report.pdf               # Full methodology
├── requirements.txt
└── README.md
```
