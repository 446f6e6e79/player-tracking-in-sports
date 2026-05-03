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


DEFAULT_WEIGHTS_PATH = Path("models/osnet_x1_0_msmt17.pt")
DEFAULT_MODEL_NAME = "osnet_x1_0"


def _pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class AppearanceEncoder:
    """OSNet-backed embedding extractor returning L2-normalized 512-D vectors."""

    EMBED_DIM = 512  # OSNet x1.0 final feature dimension

    def __init__(
        self,
        weights_path: str | Path = DEFAULT_WEIGHTS_PATH,
        model_name: str = DEFAULT_MODEL_NAME,
        device: str | torch.device | None = None,
        input_size: tuple[int, int] = (256, 128),  # OSNet's training resolution (H, W)
    ) -> None:
        try:
            from torchreid.utils import FeatureExtractor
        except ImportError as e:
            raise ImportError(
                "AppearanceEncoder requires `torchreid`. Install with:\n"
                "  pip install git+https://github.com/KaiyangZhou/deep-person-reid.git"
            ) from e

        weights_path = Path(weights_path)
        if not weights_path.exists():
            raise FileNotFoundError(
                f"OSNet weights not found at {weights_path.resolve()}.\n"
                "Download from https://kaiyangzhou.github.io/deep-person-reid/MODEL_ZOO "
                f"(file: {model_name}_msmt17_*.pth) and save it as "
                f"{weights_path}."
            )

        self.device = torch.device(device) if device is not None else _pick_device()
        self.input_size = input_size

        self._extractor = FeatureExtractor(
            model_name=model_name,
            model_path=str(weights_path),
            image_size=list(input_size),  # FeatureExtractor takes [H, W]
            device=str(self.device),
        )

    def __call__(self, frame: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        """frame: BGR HxWx3; boxes: (N, 4) xyxy. Returns (N, 512) float32."""
        if boxes is None or len(boxes) == 0:
            return np.empty((0, self.EMBED_DIM), dtype=np.float32)

        h_img, w_img = frame.shape[:2]

        crops: list[np.ndarray] = []
        keep: list[int] = []
        for i, (x1, y1, x2, y2) in enumerate(np.asarray(boxes, dtype=float)):
            x1c = int(max(0, min(w_img - 1, np.floor(x1))))
            y1c = int(max(0, min(h_img - 1, np.floor(y1))))
            x2c = int(max(0, min(w_img,     np.ceil(x2))))
            y2c = int(max(0, min(h_img,     np.ceil(y2))))
            if x2c <= x1c or y2c <= y1c:
                continue
            # FeatureExtractor builds a PIL Image from each numpy array, so it
            # expects RGB ordering.
            crops.append(cv2.cvtColor(frame[y1c:y2c, x1c:x2c], cv2.COLOR_BGR2RGB))
            keep.append(i)

        out = np.zeros((len(boxes), self.EMBED_DIM), dtype=np.float32)
        if not crops:
            return out

        with torch.inference_mode():
            feats = self._extractor(crops)               # (N, 512)
            feats = F.normalize(feats, p=2, dim=1)       # match cosine-distance assumptions
        embeds = feats.detach().cpu().numpy().astype(np.float32)

        for row, src in enumerate(keep):
            out[src] = embeds[row]
        return out
