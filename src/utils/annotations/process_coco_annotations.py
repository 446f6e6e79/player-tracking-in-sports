import json
from collections import defaultdict
from pathlib import Path


def process_coco_annotations(
    coco: dict,
    outdir: str | Path,
    fps: float = 25.0,
    source: str = "ground_truth",
) -> None:
    """
    Process COCO annotations and write per-camera JSON files compatible with our TrackingOutput schema.
    Transformations applied:
        - Splits annotations by camera, inferred from image file names (e.g. "out2_frame_0001.jpg" → "cam_2")
        - Extracts frame index as an integer (e.g. "frame_0001" → 1)
        - Normalizes frame indices to 0-based contiguous (relative to first frame in each camera)
        - Converts bounding boxes from COCO format [x, y, w, h] to corner format {x1, y1, x2, y2}
        - Maps category IDs to class names and includes confidence scores (ground truth: 1.0)
    Parameters:
        - coco (dict): loaded COCO annotations dict
        - outdir (str | Path): directory where per-camera JSON files will be written
        - fps (float): video frames per second to store in the output JSON (default: 25.0)
        - source (str): source label to store in the output JSON (default: "ground_truth")
    Returns:
        - None (writes per-camera JSON files to outdir with 0-based frame indices)
    """
    # Process the COCO data to create mappings and transform annotations into per-camera detections
    cat_name = _create_category_mapping(coco)
    image_id_to_cam, image_id_to_frame = _process_images(coco)
    detections_by_cam_frame = _process_data_format(coco, image_id_to_cam, image_id_to_frame, cat_name)

    # Write the processed detections to per-camera JSON files in the specified output directory
    _write_output_files(detections_by_cam_frame, Path(outdir), source, fps)


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
        frame_part = img["file_name"].split("frame_")[1]
        # Extract just the numeric part before the next underscore or file extension
        frame_index = int(frame_part.split("_")[0].split(".")[0])

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

        # Normalize annotation frame indices to 0-based for runtime consistency.
        frame_items = sorted(detections_by_cam_frame[cam].items())
        if not frame_items:
            continue
        first_frame_index = frame_items[0][0]

        frames = [
            {"frame_index": fi - first_frame_index, "detections": dets}
            for fi, dets in frame_items
        ]
        out = {"source": source, "camera_id": cam, "fps": fps, "frames": frames}
        with out_path.open("w") as f:
            json.dump(out, f, indent=2)
        print(
            f"{out_path}  -  {len(frames)} frames, "
            f"{sum(len(fr['detections']) for fr in frames)} detections"
        )
