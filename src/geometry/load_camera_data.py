import json
from pathlib import Path

import numpy as np

DEFAULT_CAMERA_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "camera_data"


def load_camera_data(camera_id: str, base_dir: Path | str | None = None) -> dict:
    """
    Load the calibration JSON for a specific camera.
    Parameters:
        - camera_id (str): Camera identifier, e.g. "cam_13".
        - base_dir (Path | str | None): Optional override for the directory containing
          the calibration JSON files. Defaults to <repo_root>/data/camera_data.
    Returns:
        - data (dict): Parsed JSON contents (expected keys: "mtx", "dist", "rvecs", "tvecs").
    """
    base = Path(base_dir) if base_dir is not None else DEFAULT_CAMERA_DATA_DIR
    path = base / f"{camera_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"Camera data file not found: {path}")
    with open(path, "r") as f:
        return json.load(f)


def get_intrinsics(data: dict) -> np.ndarray:
    """
    Extract the 3x3 camera intrinsic matrix from a camera-data dict.
    Parameters:
        - data (dict): Camera data as returned by load_camera_data.
    Returns:
        - mtx (np.ndarray): 3x3 camera matrix as float32.
    """
    return np.array(data["mtx"], dtype=np.float32)


def get_distortion(data: dict) -> np.ndarray:
    """
    Extract the distortion coefficients from a camera-data dict.
    Parameters:
        - data (dict): Camera data as returned by load_camera_data.
    Returns:
        - dist (np.ndarray): Distortion coefficients as float32.
    """
    return np.array(data["dist"], dtype=np.float32)


def get_extrinsics(data: dict) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract the extrinsic parameters from a camera-data dict.
    Parameters:
        - data (dict): Camera data as returned by load_camera_data.
    Returns:
        - rvecs (np.ndarray): Rotation vectors as float32.
        - tvecs (np.ndarray): Translation vectors as float32.
    """
    rvecs = np.array(data["rvecs"], dtype=np.float32)
    tvecs = np.array(data["tvecs"], dtype=np.float32)
    return rvecs, tvecs


def save_extrinsics(
    camera_id: str,
    rvec: np.ndarray,
    tvec: np.ndarray,
    base_dir: Path | str | None = None,
) -> None:
    """
    Overwrite the rvecs/tvecs entries in the camera JSON, preserving mtx and dist.
    Parameters:
        - camera_id (str): Camera identifier, e.g. "cam_13".
        - rvec (np.ndarray): Rotation vector (Rodrigues form), shape (3,) or (3, 1).
        - tvec (np.ndarray): Translation vector, shape (3,) or (3, 1).
        - base_dir (Path | str | None): Optional override for the directory containing
          the calibration JSON files. Defaults to <repo_root>/data/camera_data.
    """
    base = Path(base_dir) if base_dir is not None else DEFAULT_CAMERA_DATA_DIR
    path = base / f"{camera_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"Camera data file not found: {path}")
    with open(path, "r") as f:
        data = json.load(f)
    data["rvecs"] = np.asarray(rvec, dtype=float).reshape(3, 1).tolist()
    data["tvecs"] = np.asarray(tvec, dtype=float).reshape(3, 1).tolist()
    with open(path, "w") as f:
        json.dump(data, f)
