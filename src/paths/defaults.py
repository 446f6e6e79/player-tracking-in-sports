from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_VIDEOS_INPUT_DIR = REPO_ROOT / "data" / "videos"
DEFAULT_CAMERA_DATA_DIR  = REPO_ROOT / "data" / "camera_data"
DEFAULT_ANNOTATIONS_DIR  = REPO_ROOT / "data" / "annotations"
DEFAULT_RESULTS_DIR      = REPO_ROOT / "results"
DEFAULT_MODELS_DIR       = REPO_ROOT / "models"
