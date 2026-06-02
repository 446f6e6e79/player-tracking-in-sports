import re
from dataclasses import dataclass
from pathlib import Path

from src.paths.defaults import (
    DEFAULT_ANNOTATIONS_DIR,
    DEFAULT_CAMERA_DATA_DIR,
    DEFAULT_RESULTS_DIR,
    DEFAULT_VIDEOS_INPUT_DIR,
)

DEFAULT_ANNOTATIONS_VERSION = "tracking_01"

# Camera ids in this project follow the strict pattern `cam_<int>` (e.g. `cam_2`,
# `cam_13`). Validate at the boundary so downstream code can assume the format.
_CAMERA_ID_RE = re.compile(r"^cam_(\d+)$")


def parse_camera_id(camera_id: str) -> int:
    """Return the numeric component of a camera id like 'cam_13' → 13.

    Raises ValueError if `camera_id` does not match the expected `cam_<int>` form.
    """
    match = _CAMERA_ID_RE.match(camera_id)
    if match is None:
        raise ValueError(
            f"Invalid camera id {camera_id!r}; expected pattern 'cam_<int>' (e.g. 'cam_13')."
        )
    return int(match.group(1))


def video_filename_for_camera(camera_id: str) -> str:
    """Filename convention for the source video of a camera (`out{N}.mp4`)."""
    return f"out{parse_camera_id(camera_id)}.mp4"


@dataclass(frozen=True)
class CameraPaths:
    camera_id: str
    video: Path           # input: {videos_input_dir}/out{N}.mp4
    camera_data: Path     # input: {camera_data_dir}/{camera_id}.json
    videos_dir: Path      # output: {results_dir}/{camera_id}/videos/
    serialized_dir: Path  # output: {results_dir}/{camera_id}/serialized/

    @classmethod
    def for_camera(
        cls,
        camera_id: str,
        videos_input_dir: Path | str | None = None,
        camera_data_dir: Path | str | None = None,
        results_dir: Path | str | None = None,
    ) -> "CameraPaths":
        """Factory method to create CameraPaths for a given camera_id, with optional overrides for input and output directories."""
        vid_root = Path(videos_input_dir) if videos_input_dir is not None else DEFAULT_VIDEOS_INPUT_DIR
        cam_root = Path(camera_data_dir) if camera_data_dir is not None else DEFAULT_CAMERA_DATA_DIR
        res_root = Path(results_dir) if results_dir is not None else DEFAULT_RESULTS_DIR
        cam_dir = res_root / camera_id
        return cls(
            camera_id=camera_id,
            video=vid_root / video_filename_for_camera(camera_id),
            camera_data=cam_root / f"{camera_id}.json",
            videos_dir=cam_dir / "videos",
            serialized_dir=cam_dir / "serialized",
        )

    def mkdir(self) -> None:
        """Create the output directories for videos and serialized data if they don't already exist."""
        self.videos_dir.mkdir(parents=True, exist_ok=True)
        self.serialized_dir.mkdir(parents=True, exist_ok=True)

    def annotation_path(
        self,
        version: str = DEFAULT_ANNOTATIONS_VERSION,
        annotations_dir: Path | str | None = None,
    ) -> Path:
        """Get the path to the annotation JSON file for this camera, with optional version and directory overrides."""
        root = Path(annotations_dir) if annotations_dir is not None else DEFAULT_ANNOTATIONS_DIR
        return root / version / f"{self.camera_id}.json"

    @property
    def tracking_video(self) -> Path:
        return self.videos_dir / "tracking_resolved.mp4"

    @property
    def detection_video(self) -> Path:
        return self.videos_dir / "detection.mp4"

    @property
    def pre_resolution_video(self) -> Path:
        return self.videos_dir / "tracking_pre_resolution.mp4"

    @property
    def tracking_json(self) -> Path:
        return self.serialized_dir / "tracking.json"
