"""
Fine-tune YOLOv11m on a Roboflow YOLO-format dataset.
Designed to run on a Colab GPU.

IMPORTANT: training images must be DISJOINT from the videos used at inference time 

COLAB USAGE:
    !pip install -q ultralytics roboflow

    # from roboflow import Roboflow
    # rf = Roboflow(api_key="...")
    # ds = rf.workspace("...").project("...").version(N).download("yolov11")
    # # ds.location is then the directory containing data.yaml

    !python scripts/finetune.py \\
        --data /content/<dataset>/data.yaml \\
        --epochs 50 --device 0

NOTE: an already-wired-up Colab notebook is available in the repository as finetune.ipynb, 
which also includes instructions for downloading a Roboflow dataset. 
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
    p.add_argument("--model", default="yolo11m.pt", help="Starting weights")
    p.add_argument("--epochs", type=int, default=300, help="Number of training epochs")
    p.add_argument("--patience", type=int, default=30, help="Early stopping patience epochs with no fitness improvement")
    p.add_argument("--imgsz", type=int, default=1280, help="Train at high res so the ball is learnable")
    p.add_argument("--batch", type=int, default=4, help="Batch size")
    p.add_argument("--device", default=None, help="0 for first GPU, 'cpu' to force CPU; None=auto")
    p.add_argument("--name", default="yolo11m_basketball", help="Run name under runs/detect/")
    p.add_argument("--out", default="models/yolo11m_finetuned.pt", help="Where to copy best.pt")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    model = YOLO(args.model)
    results = model.train(
        data=args.data,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        name=args.name,
        epochs=args.epochs,
        
        # Early stopping after this many epochs with no fitness improvement (see best.pt updates);
        patience=args.patience,
        save=True,
        val=True,
        
        # Augmentations tuned for small objects;
        mosaic=1.0, mixup=0.1, hsv_v=0.4, degrees=5, translate=0.1, scale=0.5,
    )

    # `train()` return shape can vary across ultralytics versions; try common save_dir locations.
    save_dir = getattr(results, "save_dir", None) or getattr(getattr(model, "trainer", None), "save_dir", None)
    if save_dir is None:
        raise RuntimeError(
            "Could not determine training output directory (save_dir). "
            "Check ultralytics version and training logs."
        )

    save_dir = Path(save_dir)
    best = save_dir / "weights" / "best.pt"
    last = save_dir / "weights" / "last.pt"

    if not best.exists():
        if last.exists():
            print(f"Warning: best.pt not found, using last.pt from {last}")
            best = last
        else:
            raise FileNotFoundError(
                f"No checkpoint found in {save_dir / 'weights'} (missing best.pt and last.pt)."
            )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(best, out)
    print(f"\nFine-tuned model saved to: {out}")

    # Quick sanity validation on the dataset's own valid split
    YOLO(out).val(data=args.data, imgsz=args.imgsz, device=args.device)


if __name__ == "__main__":
    main()
