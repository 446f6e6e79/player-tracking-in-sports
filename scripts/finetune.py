"""Fine-tune YOLOv11m on a Roboflow YOLO-format dataset.

Designed to run on a Colab GPU. Just point it at the data.yaml that ships
inside the Roboflow export folder.

IMPORTANT: training images must be DISJOINT from the videos used at inference
time (data/videos/out{2,4,13}.mp4). Roboflow datasets ship pre-split into
train/valid/test — respect that split.

Typical Colab usage:

    !pip install -q ultralytics roboflow

    # one-time, in a Colab cell (wired by you):
    # from roboflow import Roboflow
    # rf = Roboflow(api_key="...")
    # ds = rf.workspace("...").project("...").version(N).download("yolov11")
    # # ds.location is then the directory containing data.yaml

    !python scripts/finetune.py \\
        --data /content/<dataset>/data.yaml \\
        --epochs 50 --device 0
"""

import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fine-tune YOLOv11m on a Roboflow YOLO-format dataset."
    )
    p.add_argument("--data", required=True, help="Path to data.yaml from the Roboflow export")
    p.add_argument("--model", default="models/yolo11m.pt", help="Starting weights")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--imgsz", type=int, default=1280, help="Train at high res so the ball is learnable")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--device", default=None, help="0 for first GPU, 'cpu' to force CPU; None=auto")
    p.add_argument("--name", default="yolo11m_football", help="Run name under runs/detect/")
    p.add_argument("--out", default="models/yolo11m_finetuned.pt", help="Where to copy best.pt")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    model = YOLO(args.model)
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        name=args.name,
        # Augmentations worth trying for football (motion blur + tiny ball):
        # mosaic=1.0, mixup=0.1, hsv_v=0.4, degrees=5, translate=0.1, scale=0.5,
    )

    best = Path(results.save_dir) / "weights" / "best.pt"
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(best, out)
    print(f"\nFine-tuned model saved to: {out}")

    # Quick sanity validation on the dataset's own valid split
    YOLO(out).val(data=args.data, imgsz=args.imgsz, device=args.device)


if __name__ == "__main__":
    main()
