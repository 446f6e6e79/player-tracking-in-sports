from pathlib import Path
from huggingface_hub import hf_hub_download
from ultralytics import YOLO

from src.paths.model_paths import (
    HF_REPO_ID,
    YOLO_DEFAULT_FILENAME,
    YOLO_DEFAULT_PATH,
)
from src.utils.logging import get_logger


logger = get_logger(__name__)


def load_fine_tuned_yolo_model(model_path: str | Path | None = None) -> YOLO:
    """
    Load the fine-tuned YOLOv11m basketball model from disk, downloading it from
    Hugging Face if necessary.
    Parameters:
        - model_path (str | Path): expected local path to the weights file
    """
    # If no path is provided, ensure the default model is available
    if model_path is None:
        model_path = _ensure_default_model()

    logger.info("Model ready at %s. Loading into memory...", model_path)
    return YOLO(model_path)

def _download_model_from_huggingface(
    local_dir: str | Path,
    repo_id: str = HF_REPO_ID,
    filename: str = YOLO_DEFAULT_FILENAME,
) -> Path:
    """
    Download a single weight file from a Hugging Face model repo.
    The download is idempotent: hf_hub_download checks the local cache and
    only re-downloads if the remote file has changed.
    Parameters:
        - repo_id (str): Hugging Face model repo (e.g. "user/model-name")
        - filename (str): file to download from the repo (e.g. "best.pt")
        - local_dir (str | Path): directory to materialize the file into
    Returns:
        - Path: local path to the downloaded file
    """
    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Downloading %s from Hugging Face repo '%s' into %s...",
                filename, repo_id, local_dir)
    local_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=str(local_dir),
    )
    return Path(local_path)


def _ensure_default_model(
    model_path: str | Path = YOLO_DEFAULT_PATH
) -> Path:
    """
    Return `model_path` if it already exists on disk; otherwise download the
    default best.pt from the Hugging Face repo into the same directory.
    Parameters:
        - model_path (str | Path): expected local path to the weights file
    Returns:
        - Path: local path that is guaranteed to exist
    """
    # Check if the model already exists at the given path
    model_path = Path(model_path)
    if model_path.exists():
        return model_path

    logger.info("Model not found at %s; fetching default from %s.", model_path, HF_REPO_ID)
    return _download_model_from_huggingface(
        filename=YOLO_DEFAULT_FILENAME,
        local_dir=model_path.parent,
    )
