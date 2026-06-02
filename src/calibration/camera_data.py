import json
import warnings
from dataclasses import dataclass
from pathlib import Path
import numpy as np

from src.paths.defaults import DEFAULT_CAMERA_DATA_DIR

# Frozen dataclass preventing accidental mutation of loaded camera data, and providing a single source of truth for JSON paths and load/save logic.
@dataclass(frozen=True)
class CameraData:
    """
    Structured camera data loaded from JSON, including intrinsic matrix, 
    distortion coefficients, and extrinsic parameters if available
    """
    camera_id: str
    mtx: np.ndarray   # (3, 3) intrinsic matrix
    dist: np.ndarray  # (1, k) distortion coefficients
    rvec: np.ndarray  # (3, 1) Rodrigues rotation
    tvec: np.ndarray  # (3, 1) translation
    path: Path        # JSON source path

    @classmethod
    def load(
        cls, 
        camera_id: str, 
        base_dir: Path | str = DEFAULT_CAMERA_DATA_DIR
    ) -> "CameraData":
        """Load camera data from a JSON file named `{camera_id}.json` in `base_dir`"""
        # Save the base directory as a Path object for consistent path handling
        base = Path(base_dir)
        path = base / f"{camera_id}.json"
        if not path.exists():
            raise FileNotFoundError(f"Camera data file not found: {path}")
        
        # Load the JSON data and convert to numpy arrays
        with open(path, "r") as f:
            data = json.load(f)
        zero3 = [[0.0], [0.0], [0.0]]   # Default value for rvec/tvec if not present in JSON
        # An absent rvec/tvec means the camera has not been calibrated through
        # solvePnP yet; downstream triangulation will silently produce garbage,
        # so surface a clear warning at load time.
        if "rvecs" not in data or "tvecs" not in data:
            warnings.warn(
                f"Camera {camera_id!r} has no extrinsics in {path}; defaulting rvec/tvec to zero. "
                "Run scripts/calibrate_extrinsics.py before any triangulation/3D step.",
                stacklevel=2,
            )
        return cls(
            camera_id=camera_id,
            mtx=np.array(data["mtx"], dtype=np.float32),
            dist=np.array(data["dist"], dtype=np.float32),
            rvec=np.array(data.get("rvecs", zero3), dtype=np.float32).reshape(3, 1),
            tvec=np.array(data.get("tvecs", zero3), dtype=np.float32).reshape(3, 1),
            path=path,
        )

    def save_extrinsics(
        self,
        rvec: np.ndarray,
        tvec: np.ndarray,
        landmarks_px: dict[str, tuple[float, float]] | None = None,
    ) -> None:
        """
        Overwrite the rvecs/tvecs entries in `self.path`, preserving mtx/dist.
        If `landmarks_px` is provided, also save the clicked pixel coordinates so
        that ``verify_stored_extrinsics`` can compute reprojection RMSE later.
        Does not mutate `self` (frozen dataclass) — callers reload if they need
        the updated values.
        """
        with open(self.path, "r") as f:
            data = json.load(f)
        data["rvecs"] = np.asarray(rvec, dtype=float).reshape(3, 1).tolist()
        data["tvecs"] = np.asarray(tvec, dtype=float).reshape(3, 1).tolist()
        if landmarks_px is not None:
            data["landmarks_px"] = {name: list(px) for name, px in landmarks_px.items()}
        with open(self.path, "w") as f:
            json.dump(data, f)
