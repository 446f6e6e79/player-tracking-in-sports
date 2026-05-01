import json
import shutil
import tempfile
from pathlib import Path

from roboflow import Roboflow

from src.utils.annotations.process_coco_annotations import process_coco_annotations

# Name of the COCO annotation file produced by Roboflow inside each split directory
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
    """
    Download the Roboflow annotation file for the specified dataset
    After download, if in COCO MMDetection format, the annotations are processed in a compatible way for our evaluation pipeline.
    Otherwise, the annotation file is copied as-is without processing (e.g. for YOLO format)
    Parameters:
        - workspace (str): Roboflow workspace name
        - project (str): Roboflow project name
        - version (int): dataset version number to download
        - api_key (str): Roboflow API key
        - output_dir (str | Path): directory where the annotation JSON will be written
        - split (str): dataset split to download (default: "train")
        - type (str): dataset format to download (default: "coco-mmdetection")
    Returns:
        - None (writes _annotations.coco.json to output_dir)
    """
    output_dir = Path(output_dir)

    print(f"Downloading dataset {project} v{version} from Roboflow workspace '{workspace}'...")

    # Download into a temp directory so Roboflow always gets a clean, empty location
    rf = Roboflow(api_key=api_key)
    with tempfile.TemporaryDirectory() as tmp:
        # The download method returns a Dataset object with a 'location' attribute pointing to the downloaded files
        dataset = rf.workspace(workspace).project(project).version(version).download(type, location=tmp, overwrite=True)

        # The dataset is downloaded as a directory containing the requested split (e.g. "train/") with the COCO JSON inside it.
        dataset_dir = Path(dataset.location)
        print(f"Downloaded to: {dataset_dir}")

        # Locate the COCO annotation JSON in the requested split subdirectory
        annotation_src = dataset_dir / split / _COCO_FILENAME
        if not annotation_src.exists():
            raise FileNotFoundError(
                f"Could not find {split}/{_COCO_FILENAME} inside {dataset_dir}. "
            )

        # Load and process the COCO annotations directly; temp dir is removed on context exit
        output_dir.mkdir(parents=True, exist_ok=True)
        if type == "coco-mmdetection":
            print(f"Processing COCO annotations from {annotation_src} into {output_dir}...")
            coco = _load_coco_json(annotation_src)
            process_coco_annotations(coco, output_dir)
        else:
            # For other formats, copy the annotation JSON as-is without processing
            print(f"Copying annotation file from {annotation_src} to {output_dir} without processing...")
            _copy_annotation(dataset_dir / split, output_dir)

    print("Download and processing complete.")


def _copy_annotation(split_dir: Path, output_dir: Path) -> None:
    '''
    Copy the first JSON annotation file found in split_dir to output_dir.
    Parameters:
        - split_dir (Path): directory of the downloaded split (e.g. dataset_dir / "train")
        - output_dir (Path): destination directory
    Returns:
        - None (copies the file to output_dir preserving its filename)
    '''
    # Find the first JSON file in the split directory
    src = next(split_dir.glob("*.json"), None)
    if src is None:
        raise FileNotFoundError(f"No JSON annotation file found in {split_dir}")

    # Copy to the output directory, keeping the original filename
    dst = output_dir / src.name
    shutil.copy2(src, dst)
    print(f"Copied annotation to {dst}")


def _load_coco_json(file_path: str | Path) -> dict:
    '''Load COCO annotation JSON file.'''
    with Path(file_path).open() as f:
        return json.load(f)
