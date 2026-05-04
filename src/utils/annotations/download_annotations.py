import json
import shutil
import tempfile
from pathlib import Path

from roboflow import Roboflow

from src.utils.annotations.process_coco_annotations import process_coco_annotations

_COCO_FILENAME = "_annotations.coco.json"


def download_annotations(
    workspace: str,
    project: str,
    version: int,
    api_key: str,
    output_dir: str | Path,
    split: str = "train",
    type: str = "coco-mmdetection",
) -> None:
    """Download Roboflow annotations and write per-camera JSON files to output_dir.
    COCO MMDetection format is processed into per-camera JSON with 0-based frame indices.
    Other formats are copied as-is.
    Parameters:
        - workspace: Roboflow workspace name
        - project: Roboflow project name
        - version: dataset version number
        - api_key: Roboflow API key
        - output_dir: directory where annotation JSON files will be written
        - split: dataset split to download (default: "train")
        - type: dataset format (default: "coco-mmdetection")
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {project} v{version} from '{workspace}'...")
    rf = Roboflow(api_key=api_key)

    # Roboflow's Python SDK doesn't support direct download to a specified directory, so we use a temporary directory and then move the files.
    with tempfile.TemporaryDirectory() as tmp:

        # Define the dataset type to download based on the specified format.
        dataset = rf.workspace(workspace).project(project).version(version).download(type, location=tmp, overwrite=True)
        split_dir = Path(dataset.location) / split

        # Process the downloaded annotations based on the specified format. 
        if type == "coco-mmdetection":
            # For COCO format, we need to process the annotations into per-camera JSON files with 0-based frame indices.
            annotation_src = split_dir / _COCO_FILENAME
            if not annotation_src.exists():
                raise FileNotFoundError(f"Could not find {split}/{_COCO_FILENAME} inside {dataset.location}")
            with annotation_src.open() as f:
                process_coco_annotations(json.load(f), output_dir)
        else:
            # For other formats, we assume they are already in the desired format and just copy them to the output directory.
            src = next(split_dir.glob("*.json"), None)
            if src is None:
                raise FileNotFoundError(f"No JSON annotation file found in {split_dir}")
            shutil.copy2(src, output_dir / src.name)
            print(f"Copied {src.name} to {output_dir}")

    print("Download complete.")
