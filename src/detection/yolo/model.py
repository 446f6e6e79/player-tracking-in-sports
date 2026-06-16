from pathlib import Path
from threading import Lock

from huggingface_hub import hf_hub_download
from ultralytics import YOLO

from src.paths.model_paths import (
    HF_REPO_ID,
    YOLO_DEFAULT_FILENAME,
    YOLO_DEFAULT_PATH,
)
from src.utils.logging import get_logger


logger = get_logger(__name__)

_model_cache: dict[Path, YOLO] = {}
_cache_lock = Lock()


def load_fine_tuned_yolo_model(
    model_path: str | Path | None = None,
    force_reload: bool = False,
) -> YOLO:
    """
    Load the fine-tuned YOLO26m basketball model from disk, downloading it from
    Hugging Face if necessary. Subsequent calls with the same resolved path
    return the cached instance.
    Parameters:
        - model_path (str | Path | None): expected local path to the weights file
        - force_reload (bool): bypass the cache and rebuild the YOLO instance
    """
    resolved_path = _ensure_model_available(model_path)

    with _cache_lock:
        if not force_reload and resolved_path in _model_cache:
            logger.info("Reusing cached YOLO model from %s.", resolved_path)
            return _model_cache[resolved_path]

        logger.info("Model ready at %s. Loading into memory...", resolved_path)
        model = YOLO(resolved_path)
        _model_cache[resolved_path] = model
        return model


def _ensure_model_available(model_path: str | Path | None) -> Path:
    """
    Resolve `model_path` to an existing file on disk, downloading the default
    weights from Hugging Face if the file is missing and the filename matches
    the known default.
    """
    path = Path(model_path) if model_path is not None else Path(YOLO_DEFAULT_PATH)

    if path.exists():
        return path.resolve()

    if path.name != YOLO_DEFAULT_FILENAME:
        raise FileNotFoundError(
            f"Model not found at {path} and filename does not match the "
            f"downloadable default '{YOLO_DEFAULT_FILENAME}'."
        )

    logger.info("Model not found at %s; fetching default from %s.", path, HF_REPO_ID)
    return _download_model_from_huggingface(
        filename=YOLO_DEFAULT_FILENAME,
        local_dir=path.parent,
    ).resolve()


def _download_model_from_huggingface(
    local_dir: str | Path,
    repo_id: str = HF_REPO_ID,
    filename: str = YOLO_DEFAULT_FILENAME,
) -> Path:
    """
    Download a single weight file from a Hugging Face model repo.
    The download is idempotent: hf_hub_download checks the local cache and
    only re-downloads if the remote file has changed. If the download fails
    partway, any partial file at the target path is removed before re-raising.
    """
    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)
    target = local_dir / filename

    logger.info("Downloading %s from Hugging Face repo '%s' into %s...",
                filename, repo_id, local_dir)
    try:
        local_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=str(local_dir),
        )
    except Exception:
        if target.exists():
            logger.warning("Removing partial download at %s.", target)
            target.unlink(missing_ok=True)
        raise

    return Path(local_path)
