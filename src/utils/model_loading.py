from pathlib import Path

from huggingface_hub import hf_hub_download
from ultralytics import YOLO

# Hugging Face repo where the fine-tuned YOLOv11m basketball weights are published.
HF_REPO_ID = "446f6e6e79/yolo11m-basketball-fineTuned"
DEFAULT_FILENAME = "best.pt"
DEFAULT_LOCAL_DIR = Path("models/fine_tuned_models")


def load_fine_tuned_yolo_model(model_path: str | Path | None = None) -> YOLO:
    """
    Load the fine-tuned YOLOv11m basketball model from disk, downloading it from
    Hugging Face if necessary.
    Parameters:
        - model_path (str | Path): expected local path to the weights file
    """
    model_path = ensure_default_model(model_path)
    print(f"Model ready at {model_path}. Loading into memory...")
    return YOLO(model_path)

def download_model_from_huggingface(
    repo_id: str = HF_REPO_ID,
    filename: str = DEFAULT_FILENAME,
    local_dir: str | Path = DEFAULT_LOCAL_DIR,
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

    print(f"Downloading {filename} from Hugging Face repo '{repo_id}' into {local_dir}...")
    local_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=str(local_dir),
    )
    return Path(local_path)


def ensure_default_model(model_path: str | Path | None = None) -> Path:
    """
    Return `model_path` if it already exists on disk; otherwise download the
    default best.pt from the Hugging Face repo into the same directory.
    Parameters:
        - model_path (str | Path | None): expected local path to the weights file
    Returns:
        - Path: local path that is guaranteed to exist
    """
    if model_path is None:
        model_path = DEFAULT_LOCAL_DIR / DEFAULT_FILENAME

    model_path = Path(model_path)
    if model_path.exists():
        return model_path

    print(f"Model not found at {model_path}; fetching default from {HF_REPO_ID}.")
    return download_model_from_huggingface(
        filename=DEFAULT_FILENAME,
        local_dir=model_path.parent,
    )
