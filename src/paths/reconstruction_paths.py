from dataclasses import dataclass
from pathlib import Path

from src.paths.defaults import DEFAULT_RESULTS_DIR


@dataclass(frozen=True)
class ReconstructionPaths:
    camera_ids: tuple[str, ...]
    videos_dir: Path      # output: {results_dir}/reconstruction/{key}/videos/
    serialized_dir: Path  # output: {results_dir}/reconstruction/{key}/serialized/

    @classmethod
    def for_cameras(
        cls,
        camera_ids: tuple[str, ...],
        results_dir: Path | str | None = None,
    ) -> "ReconstructionPaths":
        res_root = Path(results_dir) if results_dir is not None else DEFAULT_RESULTS_DIR
        key = "__".join(camera_ids)
        recon_dir = res_root / "reconstruction" / key
        return cls(
            camera_ids=camera_ids,
            videos_dir=recon_dir / "videos",
            serialized_dir=recon_dir / "serialized",
        )

    def mkdir(self) -> None:
        self.videos_dir.mkdir(parents=True, exist_ok=True)
        self.serialized_dir.mkdir(parents=True, exist_ok=True)

    @property
    def triangulation_json(self) -> Path:
        return self.serialized_dir / "triangulation.json"

    @property
    def minimap_video(self) -> Path:
        return self.videos_dir / "minimap.mp4"

    @property
    def overlay_video(self) -> Path:
        return self.videos_dir / "overlay.mp4"

    @property
    def scene_3d_video(self) -> Path:
        return self.videos_dir / "scene_3d.mp4"
