import os
from pathlib import Path

# Point urllib at certifi's CA bundle. Without this, the macOS Python.app
# distribution falls back to a trust store that fails CDN cert validation.
import certifi
os.environ.setdefault("SSL_CERT_FILE", certifi.where())
os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from src.tracking.sort_components.osnet import (
    osnet_x0_25,
    osnet_x0_5,
    osnet_x0_75,
    osnet_x1_0,
    osnet_ibn_x1_0,
)

# Default variables for AppearanceEncoder.
# The weights file is not included in the repo due to size, but can be downloaded from the official OSNet model zoo.
DEFAULT_WEIGHTS_PATH = Path("models/osnet_x1_0_msmt17.pt")
DEFAULT_MODEL_NAME = "osnet_x1_0"

# Vendored OSNet variants — same factory signatures as torchreid.models.
_OSNET_FACTORIES = {
    "osnet_x1_0": osnet_x1_0,
    "osnet_x0_75": osnet_x0_75,
    "osnet_x0_5": osnet_x0_5,
    "osnet_x0_25": osnet_x0_25,
    "osnet_ibn_x1_0": osnet_ibn_x1_0,
}

# ImageNet stats — the preprocessing torchreid's FeatureExtractor used. Keeping
# these identical preserves embedding compatibility with previously-tuned
# cosine-distance thresholds in DeepSORT.
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def _pick_device() -> torch.device:
    """Helper to select the best available device for PyTorch computations. Prefers CUDA, then MPS (Apple Silicon), then CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _load_osnet_weights(model: torch.nn.Module, weights_path: Path) -> None:
    """Load OSNet checkpoint into model, ignoring keys that don't match by name+shape.

    Mirrors torchreid.utils.load_pretrained_weights so that .pt files from the
    deep-person-reid model zoo load cleanly. The classifier head is expected to
    mismatch (we never use it for inference) and is silently discarded.
    """
    checkpoint = torch.load(str(weights_path), map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint

    model_dict = model.state_dict()
    matched = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            k = k[7:]
        if k in model_dict and model_dict[k].shape == v.shape:
            matched[k] = v
    if not matched:
        raise RuntimeError(
            f"No weight tensors in {weights_path} matched the OSNet model by name+shape. "
            "Is this the right checkpoint for the chosen model_name?"
        )
    model_dict.update(matched)
    model.load_state_dict(model_dict)


class AppearanceEncoder:
    """
    AppearanceEncoder uses a pre-trained OSNet model to extract 512-dimensional feature vectors
    for person re-identification.
    The features are L2-normalized to be compatible with cosine distance-based matching in DeepSORT.
    Parameters:
        - weights_path: Path to the OSNet weights file (e.g., osnet_x1_0_msmt17.pt).
        - model_name: The specific OSNet architecture to use.
        - device: The torch device to run the model on (e.g., "cuda", "mps", or "cpu"). If None, it will be auto-selected.
        - input_size: The (height, width) to which detected person crops will be resized before feature extraction. OSNet was trained on 256x128 crops.
    Usage:
    """
    EMBED_DIM = 512  # OSNet x1.0 final feature dimension

    def __init__(
        self,
        weights_path: str | Path = DEFAULT_WEIGHTS_PATH,
        model_name: str = DEFAULT_MODEL_NAME,
        device: str | torch.device | None = None,
        input_size: tuple[int, int] = (256, 128),  # OSNet's training resolution (H, W)
    ) -> None:
        if model_name not in _OSNET_FACTORIES:
            raise ValueError(
                f"Unknown model_name {model_name!r}. Expected one of {sorted(_OSNET_FACTORIES)}."
            )

        # Check that the weights file exists, and provide instructions if it doesn't.
        weights_path = Path(weights_path)
        if not weights_path.exists():
            raise FileNotFoundError(
                f"OSNet weights not found at {weights_path.resolve()}.\n"
                "Download from https://kaiyangzhou.github.io/deep-person-reid/MODEL_ZOO "
                f"(file: {model_name}_msmt17_*.pth) and save it as "
                f"{weights_path}."
            )

        # If no device was specified, pick the best available one.
        self.device = torch.device(device) if device is not None else _pick_device()
        self.input_size = input_size

        # Build the vendored OSNet, load weights from disk, freeze for inference.
        model = _OSNET_FACTORIES[model_name](num_classes=1000)
        _load_osnet_weights(model, weights_path)
        model.eval()
        model.to(self.device)
        self._model = model

        # Pre-build broadcastable normalization tensors on the target device so
        # we don't rebuild them on every call.
        self._mean = torch.tensor(_IMAGENET_MEAN, device=self.device).view(1, 3, 1, 1)
        self._std = torch.tensor(_IMAGENET_STD, device=self.device).view(1, 3, 1, 1)

    def __call__(self, frame: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        """Extract appearance features for the given frame and bounding boxes.
        Parameters:
            - frame: The current video frame as a BGR numpy array (H, W, 3).
            - boxes: An array of bounding boxes in (x1, y1, x2, y2) format, shape (N, 4).
        Returns:
            - A (N, EMBED_DIM) array of L2-normalized feature vectors corresponding to each bounding box."""
        # If no boxes are provided, return an empty array with the correct shape.
        if boxes is None or len(boxes) == 0:
            return np.empty((0, self.EMBED_DIM), dtype=np.float32)

        h_img, w_img = frame.shape[:2]
        crops: list[np.ndarray] = []
        keep: list[int] = []

        # For each bounding box, extract valid int coordinates, clip to image boundaries
        for i, (x1, y1, x2, y2) in enumerate(np.asarray(boxes, dtype=float)):
            x1c = int(max(0, min(w_img - 1, np.floor(x1))))
            y1c = int(max(0, min(h_img - 1, np.floor(y1))))
            x2c = int(max(0, min(w_img,     np.ceil(x2))))
            y2c = int(max(0, min(h_img,     np.ceil(y2))))
            
            # Skip invalid boxes that have non-positive width or height after clipping.
            if x2c <= x1c or y2c <= y1c:
                continue
            
            # Extract the BBOX crop from the frame, convert from BGR to RGB, and store it for batch processing.
            crops.append(cv2.cvtColor(frame[y1c:y2c, x1c:x2c], cv2.COLOR_BGR2RGB))
            keep.append(i)

        # Initialize the output array with zeros
        out = np.zeros((len(boxes), self.EMBED_DIM), dtype=np.float32)
        # If no valid crops were extracted, return the zero array
        if not crops:
            return out

        # Resize each RGB crop to the model's input resolution, stack into an
        # (N, H, W, 3) uint8 array, then move to the device as a normalized
        # float tensor in NCHW form.
        h_in, w_in = self.input_size
        resized = np.stack(
            [cv2.resize(c, (w_in, h_in), interpolation=cv2.INTER_LINEAR) for c in crops],
            axis=0,
        )
        batch = torch.from_numpy(resized).to(self.device)
        batch = batch.permute(0, 3, 1, 2).float().div_(255.0)
        batch = (batch - self._mean) / self._std

        # OSNet in eval() returns the 512-d feature vector (pre-classifier).
        with torch.inference_mode():
            features = self._model(batch)                       # (N, 512)
            features = F.normalize(features, p=2, dim=1)        # Normalize to unit length for cosine distance compatibility
        embeds = features.detach().cpu().numpy().astype(np.float32)

        for row, src in enumerate(keep):
            out[src] = embeds[row]
        return out
