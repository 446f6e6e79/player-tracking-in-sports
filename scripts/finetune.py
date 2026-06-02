"""
Fine-tune YOLOv26m on a Roboflow YOLO-format dataset.
Designed to run on a Colab GPU.

IMPORTANT: training images must be DISJOINT from the videos used at inference time 

COLAB USAGE:
    !pip install -q ultralytics roboflow

    # from roboflow import Roboflow
    # rf = Roboflow(api_key="...")
    # ds = rf.workspace("...").project("...").version(N).download("yolov26")

    # # ds.location is then the directory containing data.yaml

    !python scripts/finetune.py \\
        --data /content/<dataset>/data.yaml \\
        --epochs 50 --device 0

NOTE: an already-wired-up Colab notebook is available in the repository as notebooks/finetune.ipynb,
which also includes instructions for downloading a Roboflow dataset.
"""

import argparse
import shutil
import sys
from pathlib import Path

from ultralytics import YOLO

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.config import get_config
from src.utils.logging import configure_logging, get_logger


logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fine-tune YOLOv26m on a Roboflow YOLO-format dataset."
    )
    p.add_argument("--data", required=True, help="Path to data.yaml from the Roboflow export")
    p.add_argument("--model", default=None, help="Starting weights (default: from config)")
    p.add_argument("--epochs", type=int, default=None, help="Number of training epochs (default: from config)")
    p.add_argument("--patience", type=int, default=None, help="Early stopping patience epochs (default: from config)")
    p.add_argument("--imgsz", type=int, default=None, help="Training image size (default: from config)")
    p.add_argument("--batch", type=int, default=None, help="Batch size (default: from config)")
    p.add_argument("--device", default=None, help="0 for first GPU, 'cpu' to force CPU; None=auto")
    p.add_argument("--name", default=None, help="Run name under runs/detect/ (default: from config)")
    p.add_argument("--out", default=None, help="Where to copy best.pt (default: from config)")
    return p.parse_args()


def main() -> None:
    configure_logging()
    args = parse_args()
    cfg = get_config().training

    # CLI flags override config values when provided.
    run_model   = args.model   or cfg.model
    run_epochs  = args.epochs  if args.epochs  is not None else cfg.epochs
    run_patience= args.patience if args.patience is not None else cfg.patience
    run_imgsz   = args.imgsz   if args.imgsz   is not None else cfg.imgsz
    run_batch   = args.batch   if args.batch   is not None else cfg.batch
    run_name    = args.name    or cfg.name
    run_out     = args.out     or cfg.out

    aug = cfg.augmentation
    model = YOLO(run_model)
    results = model.train(
        data=args.data,
        imgsz=run_imgsz,
        batch=run_batch,
        device=args.device,
        name=run_name,
        epochs=run_epochs,
        patience=run_patience,
        save=True,
        val=True,
        mosaic=aug.mosaic,
        mixup=aug.mixup,
        hsv_v=aug.hsv_v,
        degrees=aug.degrees,
        translate=aug.translate,
        scale=aug.scale,
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
            logger.warning("best.pt not found, using last.pt from %s", last)
            best = last
        else:
            raise FileNotFoundError(
                f"No checkpoint found in {save_dir / 'weights'} (missing best.pt and last.pt)."
            )

    out = Path(run_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(best, out)
    logger.info("Fine-tuned model saved to: %s", out)

    # Quick sanity validation on the dataset's own valid split
    YOLO(out).val(data=args.data, imgsz=run_imgsz, device=args.device)


if __name__ == "__main__":
    main()
