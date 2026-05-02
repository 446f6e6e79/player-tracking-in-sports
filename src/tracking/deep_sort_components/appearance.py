"""Appearance feature extractor for DeepSORT re-identification.

Wraps a torchvision ResNet18 (ImageNet-pretrained) with the classifier head
replaced by global avg pool + L2 normalize. Given a frame and a set of
detection boxes, returns one 512-D embedding per box.

This is a deliberate baseline: ImageNet features are not as discriminative as
a real ReID network trained on person crops (e.g. OSNet, FastReID), but they
are good enough to break IoU-only ID-swap ties during occlusions and they
require no extra weight files to ship.
"""
from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights


def _pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class AppearanceEncoder:
    """ResNet18-backed embedding extractor returning L2-normalized 512-D vectors."""

    EMBED_DIM = 512

    def __init__(
        self,
        device: str | torch.device | None = None,
        input_size: tuple[int, int] = (128, 64),  # (H, W) — ReID convention
    ) -> None:
        self.device = torch.device(device) if device is not None else _pick_device()
        self.input_size = input_size

        backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        # Drop the original avg pool + fc head; keep conv stages only.
        self.feature_extractor = (
            nn.Sequential(*list(backbone.children())[:-2]).to(self.device).eval()
        )
        self._mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        self._std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)

    def __call__(self, frame: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        """frame: BGR HxWx3; boxes: (N, 4) xyxy. Returns (N, 512) float32."""
        if boxes is None or len(boxes) == 0:
            return np.empty((0, self.EMBED_DIM), dtype=np.float32)

        h_img, w_img = frame.shape[:2]
        target_h, target_w = self.input_size

        crops: list[np.ndarray] = []
        keep: list[int] = []
        for i, (x1, y1, x2, y2) in enumerate(np.asarray(boxes, dtype=float)):
            x1c = int(max(0, min(w_img - 1, np.floor(x1))))
            y1c = int(max(0, min(h_img - 1, np.floor(y1))))
            x2c = int(max(0, min(w_img,     np.ceil(x2))))
            y2c = int(max(0, min(h_img,     np.ceil(y2))))
            if x2c <= x1c or y2c <= y1c:
                continue
            crops.append(frame[y1c:y2c, x1c:x2c])
            keep.append(i)

        out = np.zeros((len(boxes), self.EMBED_DIM), dtype=np.float32)
        if not crops:
            return out

        resized = np.stack(
            [
                cv2.resize(
                    cv2.cvtColor(c, cv2.COLOR_BGR2RGB), (target_w, target_h)
                ).astype(np.float32)
                / 255.0
                for c in crops
            ],
            axis=0,
        )  # (N, H, W, 3)
        batch = torch.from_numpy(resized).permute(0, 3, 1, 2).contiguous().to(self.device)
        batch = (batch - self._mean) / self._std

        with torch.inference_mode():
            feats = self.feature_extractor(batch)               # (N, 512, h', w')
            feats = F.adaptive_avg_pool2d(feats, 1).flatten(1)  # (N, 512)
            feats = F.normalize(feats, p=2, dim=1)

        embeds = feats.detach().cpu().numpy().astype(np.float32)
        for row, src in enumerate(keep):
            out[src] = embeds[row]
        return out
