import json
import os
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv
from roboflow import Roboflow

# Name of the COCO annotation file produced by Roboflow inside each split directory
_COCO_FILENAME = "_annotations.coco.json"


def download_annotations(
    workspace: str,
    project: str,
    version: int,
    output_dir: str | Path,
    split: str = "train",
    type: str = "coco-mmdetection",
) -> None:
    """
    Download the Roboflow COCO annotation JSON into output_dir.

    Skips the download entirely if any JSON file already exists in output_dir.
    After download, extracts only the COCO JSON and removes the full dataset directory.
    Parameters:
        - workspace (str): Roboflow workspace name
        - project (str): Roboflow project name
        - version (int): dataset version number to download
        - output_dir (str | Path): directory where the annotation JSON will be written
        - split (str): dataset split to download (default: "train")
        - type (str): dataset format to download (default: "coco-mmdetection")
    Returns:
        - None (writes _annotations.coco.json to output_dir)
    """
    output_dir = Path(output_dir)

    # Skip if any JSON is already present in the output directory
    existing_json = list(output_dir.glob("*.json"))
    if existing_json:
        print(f"Annotations already present in {output_dir} — skipping download.")
        return

    # Resolve API key from env if not passed explicitly
    load_dotenv()
    api_key = os.environ.get("ANNOTATIONS_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "ANNOTATIONS_API_KEY is not set. Add it to your .env file or pass it explicitly."
        )

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

def process_coco_annotations(
    coco: dict,
    outdir: str | Path,
    fps: float = 25.0,
    source: str = "ground_truth",
) -> None:
    """
    Split a COCO annotations dict into per-camera TrackingOutput JSON files.

    Transformations applied:
        - Splits annotations by camera, inferred from image file names (e.g. "out2_frame_0001.jpg" → "cam_2")
        - Extracts frame index as an integer (e.g. "frame_0001" → 1)
        - Converts bounding boxes from COCO format [x, y, w, h] to corner format {x1, y1, x2, y2}
        - Maps category IDs to class names and includes confidence scores (ground truth: 1.0)
    Parameters:
        - coco (dict): loaded COCO annotations dict
        - outdir (str | Path): directory where per-camera JSON files will be written
        - fps (float): video frames per second to store in the output JSON (default: 25.0)
        - source (str): source label to store in the output JSON (default: "ground_truth")
    Returns:
        - None (writes per-camera JSON files to outdir)
    """
    # Process the COCO data to create mappings and transform annotations into per-camera detections
    cat_name = _create_category_mapping(coco)
    image_id_to_cam, image_id_to_frame = _process_images(coco)
    detections_by_cam_frame = _process_data_format(coco, image_id_to_cam, image_id_to_frame, cat_name)

    # Write the processed detections to per-camera JSON files in the specified output directory
    _write_output_files(detections_by_cam_frame, Path(outdir), source, fps)


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


def _camera_from_filename(file_name: str) -> str | None:
    '''
    Extract camera name from the image file name, which is expected to start with "out{N}_".
    Parameters:
        - file_name (str): the file name of the image, e.g. "out2_frame_0001.jpg"
    Returns:
        - camera name in the format "cam_{N}", e.g. "cam_2", or None if the prefix is unrecognised
    '''
    prefix = file_name.split("_")[0]

    # Sanity check that the prefix starts with "out" and extract the camera number
    if not prefix.startswith("out"):
        print(f"WARNING: unrecognised prefix '{prefix}' in file name '{file_name}'")
        return None

    # Convert "out{N}" to "cam_{N}"
    return "cam_" + prefix[len("out"):]       # "cam_2", "cam_13"


def _create_category_mapping(coco: dict) -> dict[int, str]:
    '''Create mapping from category ID to category name.'''
    return {c["id"]: c["name"] for c in coco["categories"]}


def _process_images(coco: dict) -> tuple[dict[int, str], dict[int, int]]:
    '''
    Extract camera and frame index for each image.
    Parameters:
        - coco (dict): the loaded COCO JSON data
    Returns:
        - image_id_to_cam: mapping from image ID to camera name
        - image_id_to_frame: mapping from image ID to frame index
    '''
    image_id_to_cam: dict[int, str]   = {}
    image_id_to_frame: dict[int, int] = {}

    # Process each image in the COCO dataset
    for img in coco["images"]:
        # Infer camera from file name
        cam = _camera_from_filename(img["file_name"])
        if cam is None:
            continue

        # Extract frame index from file name, which is expected to contain "frame_{index}"
        frame_index = int(img["file_name"].split("frame_")[1].split("_")[0])

        # Store mappings
        image_id_to_cam[img["id"]]   = cam
        image_id_to_frame[img["id"]] = frame_index
    return image_id_to_cam, image_id_to_frame


def _process_data_format(
    coco: dict,
    image_id_to_cam: dict[int, str],
    image_id_to_frame: dict[int, int],
    cat_name: dict[int, str],
) -> dict[str, dict[int, list]]:
    '''
    Transform annotations into detections grouped by camera and frame.
    Parameters:
        - coco (dict): the loaded COCO JSON data
        - image_id_to_cam (dict): mapping from image ID to camera name
        - image_id_to_frame (dict): mapping from image ID to frame index
        - cat_name (dict): mapping from category ID to category name
    Returns:
        - detections_by_cam_frame[cam][frame_index] = [Detection dicts]
    '''
    detections_by_cam_frame: dict[str, dict[int, list]] = defaultdict(lambda: defaultdict(list))

    # Process each annotation in the COCO dataset
    for ann in coco["annotations"]:

        # Get the image ID for this annotation and find the corresponding camera and frame
        img_id = ann["image_id"]
        cam    = image_id_to_cam.get(img_id)
        if cam is None:
            continue

        # Convert COCO bbox [x, y, w, h] to {"x1": x, "y1": y, "x2": x+w, "y2": y+h}
        x, y, w, h = ann["bbox"]
        detection = {
            "bbox":       {"x1": x, "y1": y, "x2": x + w, "y2": y + h},     # Convert to corner format
            "confidence": 1.0,                                              # Ground truth annotations have confidence 1.0
            "class_id":   ann["category_id"],                               # Category ID from COCO annotation
            "class_name": cat_name[ann["category_id"]],                     # Map category ID to category name
            "track_id":   None,                                             # No track ID for detections
        }

        # Append the detection to the appropriate camera and frame in the output structure
        detections_by_cam_frame[cam][image_id_to_frame[img_id]].append(detection)
    return detections_by_cam_frame


def _write_output_files(
    detections_by_cam_frame: dict[str, dict[int, list]],
    outdir: Path,
    source: str,
    fps: float,
) -> None:
    '''
    Write per-camera detection files as JSON.
    Parameters:
        - detections_by_cam_frame (dict): nested dict of detections grouped by camera and frame
        - outdir (Path): directory where output JSON files will be written
        - source (str): source label to store in the output JSON
        - fps (float): video frames per second to store in the output JSON
    Returns:
        - None (writes files to disk)
    '''
    for cam in sorted(detections_by_cam_frame):
        out_path = outdir / f"{cam}.json"
        if out_path.exists():
            print(f"{out_path}  skipped (already exists)")
            continue
        frames = [
            {"frame_index": fi, "detections": dets}
            for fi, dets in sorted(detections_by_cam_frame[cam].items())
        ]
        out = {"source": source, "camera_id": cam, "fps": fps, "frames": frames}
        with out_path.open("w") as f:
            json.dump(out, f, indent=2)
        print(
            f"{out_path}  -  {len(frames)} frames, "
            f"{sum(len(fr['detections']) for fr in frames)} detections"
        )


