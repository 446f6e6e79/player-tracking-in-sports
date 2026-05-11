from src.paths.defaults import DEFAULT_MODELS_DIR

# Fine-tuned YOLO (auto-downloaded from Hugging Face on first use)
HF_REPO_ID            = "446f6e6e79/YOLO-basketball-fineTuned"
YOLO_DEFAULT_FILENAME = "best.pt"
YOLO_FINE_TUNED_DIR   = DEFAULT_MODELS_DIR / "fine_tuned_models"
YOLO_DEFAULT_PATH     = YOLO_FINE_TUNED_DIR / YOLO_DEFAULT_FILENAME

# OSNet appearance encoder weights (manual download from deep-person-reid model zoo)
OSNET_WEIGHTS_PATH    = DEFAULT_MODELS_DIR / "osnet_x1_0_msmt17.pt"
