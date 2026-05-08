import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DEFAULT_CAMERA_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "camera_data"

# Frozen dataclass preventing accidental mutation of loaded camera data, and providing a single source of truth for JSON paths and load/save logic.
@dataclass(frozen=True)
class CameraData:
    """Calibration parameters for a single camera, parsed from its JSON file."""
    camera_id: str
    mtx: np.ndarray   # (3, 3) intrinsic matrix
    dist: np.ndarray  # (1, k) distortion coefficients
    rvec: np.ndarray  # (3, 1) Rodrigues rotation
    tvec: np.ndarray  # (3, 1) translation
    path: Path        # JSON source path

    @classmethod
    def load(cls, camera_id: str, base_dir: Path | str | None = None) -> "CameraData":
        """Parse `<base_dir>/<camera_id>.json`. `base_dir` defaults to <repo_root>/data/camera_data."""
        base = Path(base_dir) if base_dir is not None else DEFAULT_CAMERA_DATA_DIR
        path = base / f"{camera_id}.json"
        if not path.exists():
            raise FileNotFoundError(f"Camera data file not found: {path}")
        with open(path, "r") as f:
            data = json.load(f)
        zero3 = [[0.0], [0.0], [0.0]]
        return cls(
            camera_id=camera_id,
            mtx=np.array(data["mtx"], dtype=np.float32),
            dist=np.array(data["dist"], dtype=np.float32),
            rvec=np.array(data.get("rvecs", zero3), dtype=np.float32).reshape(3, 1),
            tvec=np.array(data.get("tvecs", zero3), dtype=np.float32).reshape(3, 1),
            path=path,
        )

    def save_extrinsics(self, rvec: np.ndarray, tvec: np.ndarray) -> None:
        """
        Overwrite the rvecs/tvecs entries in `self.path`, preserving mtx/dist.
        Does not mutate `self` (frozen dataclass) — callers reload if they need
        the updated values.
        """
        with open(self.path, "r") as f:
            data = json.load(f)
        data["rvecs"] = np.asarray(rvec, dtype=float).reshape(3, 1).tolist()
        data["tvecs"] = np.asarray(tvec, dtype=float).reshape(3, 1).tolist()
        with open(self.path, "w") as f:
            json.dump(data, f)
