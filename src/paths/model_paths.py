from src.config import get_config
from src.paths.defaults import DEFAULT_MODELS_DIR

_m = get_config().models

HF_REPO_ID            = _m.hf_repo_id
YOLO_DEFAULT_FILENAME = _m.yolo_filename
YOLO_FINE_TUNED_DIR   = DEFAULT_MODELS_DIR / "fine_tuned_models"
YOLO_DEFAULT_PATH     = YOLO_FINE_TUNED_DIR / YOLO_DEFAULT_FILENAME

# OSNet appearance encoder weights (manual download from deep-person-reid model zoo)
OSNET_WEIGHTS_PATH    = DEFAULT_MODELS_DIR / _m.osnet_filename
