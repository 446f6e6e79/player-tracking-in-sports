"""Convert a Roboflow COCO JSON file into per-camera TrackingOutput JSON files.

This script processes COCO-format annotations and transforms them into a per-camera detection format
that matches src/tracking/schema.py. Each output file contains all detections for a single camera.

Transformations applied:
- Splits annotations by camera, inferred from image file names (e.g. "out2_frame_0001.jpg" - "cam_2")
- Extracts frame index as an integer (e.g. "frame_0001" - 1)
- Converts bounding boxes from COCO format [x, y, w, h] to corner format {"x1": x, "y1": y, "x2": x+w, "y2": y+h}
- Maps category IDs to class names and includes confidence scores (ground truth: 1.0)

Usage:
    python scripts/annotations.py
    python scripts/annotations.py --input PATH --outdir DIR --fps 25.0
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    '''
    Parse command line arguments for the annotation splitting script.
    Returns:
    - Namespace with the following attributes:
        - input: path to the source COCO JSON file (default: "data/annotations/_annotations.coco.json")
        - outdir: directory where per-camera JSON files will be written (default: "data/annotations")
        - fps: video frames per second to store in the output JSON (default: 25.0)
        - source: source label to store in the output JSON (default: "ground_truth")
    '''
    p = argparse.ArgumentParser(description="Split COCO annotations by camera.")
    p.add_argument(
        "--input",
        default="data/annotations/_annotations.coco.json",
        help="Source COCO JSON file",
    )
    p.add_argument(
        "--outdir",
        default="data/annotations",
        help="Directory where per-camera files are written",
    )
    p.add_argument("--fps",    type=float, default=25.0, help="Video FPS stored in TrackingOutput")
    p.add_argument("--source", default="ground_truth",   help="Source label stored in TrackingOutput")
    return p.parse_args()


def load_coco_json(file_path: str) -> dict:
    '''Load COCO annotation JSON file.'''
    with Path(file_path).open() as f:
        return json.load(f)
    

def camera_from_filename(file_name: str) -> str | None:
    '''
    Extract camera name from the image file name, which is expected to start with "out{N}_".
    Parameter:
    - file_name: the file name of the image, e.g. "out2_frame_0001.jpg"
    Returns:
    - camera name in the format "cam_{N}", e.g. "cam_2"
    '''
    prefix = file_name.split("_")[0]          

    # Sanity check that the prefix starts with "out" and extract the camera number
    if not prefix.startswith("out"):
        print(f"WARNING: unrecognised prefix '{prefix}' in file name '{file_name}'")
        return None
    
    # Convert "out{N}" to "cam_{N}"
    return "cam_" + prefix[len("out"):]       # "cam_2", "cam_13"


def create_category_mapping(coco: dict) -> dict[int, str]:
    '''Create mapping from category ID to category name.'''
    return {c["id"]: c["name"] for c in coco["categories"]}


def process_images(coco: dict) -> tuple[dict[int, str], dict[int, int]]:
    '''
    Extract camera and frame index for each image.
    Parameters:
        - coco: the loaded COCO JSON data
    Returns:
        - image_id_to_cam: mapping from image ID to camera name
        - image_id_to_frame: mapping from image ID to frame index
    '''
    image_id_to_cam: dict[int, str] = {}
    image_id_to_frame: dict[int, int] = {}

    # Process each image in the COCO dataset
    for img in coco["images"]:
        # Infer camera from file name
        cam = camera_from_filename(img["file_name"])
        if cam is None:
            continue

        # Extract frame index from file name, which is expected to contain "frame_{index}"
        frame_index = int(img["file_name"].split("frame_")[1].split("_")[0])

        # Store mappings
        image_id_to_cam[img["id"]] = cam
        image_id_to_frame[img["id"]] = frame_index
    return image_id_to_cam, image_id_to_frame


def process_annotations(
    coco: dict,
    image_id_to_cam: dict[int, str],
    image_id_to_frame: dict[int, int],
    cat_name: dict[int, str],
) -> dict[str, dict[int, list]]:
    '''
    Transform annotations into detections grouped by camera and frame.
    Parameters:
        - coco: the loaded COCO JSON data
        - image_id_to_cam: mapping from image ID to camera name
        - image_id_to_frame: mapping from image ID to frame index
        - cat_name: mapping from category ID to category name
    Returns:
        - detections_by_cam_frame[cam][frame_index] = [Detection dicts]
    '''
    detections_by_cam_frame: dict[str, dict[int, list]] = defaultdict(lambda: defaultdict(list))

    # Process each annotation in the COCO dataset
    for ann in coco["annotations"]:

        # Get the image ID for this annotation and find the corresponding camera and frame
        img_id = ann["image_id"]
        cam = image_id_to_cam.get(img_id)
        if cam is None:
            continue

        # Convert COCO bbox [x, y, w, h] to {"x1": x, "y1": y, "x2": x+w, "y2": y+h}
        x, y, w, h = ann["bbox"]
        detection = {
            "bbox": {"x1": x, "y1": y, "x2": x + w, "y2": y + h},   # Convert to corner format
            "confidence": 1.0,                                      # Ground truth annotations have confidence 1.0               
            "class_id": ann["category_id"],                         # Category ID from COCO annotation
            "class_name": cat_name[ann["category_id"]],             # Map category ID to category name using the provided mapping
            "track_id": None,                                       # No track ID for detections   
        }

        # Append the detection to the appropriate camera and frame in the output structure
        detections_by_cam_frame[cam][image_id_to_frame[img_id]].append(detection)
    return detections_by_cam_frame


def write_output_files(
    detections_by_cam_frame: dict[str, dict[int, list]],
    outdir: Path,
    source: str,
    fps: float,
) -> None:
    '''
    Write per-camera detection files as JSON.
    Parameters:
        - detections_by_cam_frame: nested dict of detections grouped by camera and frame
        - outdir: directory where output JSON files will be written
        - source: source label to store in the output JSON
        - fps: video frames per second to store in the output JSON  
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
            f"{sum(len(f['detections']) for f in frames)} detections"
        )


''' Main function to orchestrate the annotation processing workflow. '''
def main() -> None:
    args = parse_args()

    # Load the COCO JSON annotations
    coco = load_coco_json(args.input)

    # Process the COCO data to create mappings and transform annotations into per-camera detections
    cat_name = create_category_mapping(coco)
    image_id_to_cam, image_id_to_frame = process_images(coco)
    detections_by_cam_frame = process_annotations(coco, image_id_to_cam, image_id_to_frame, cat_name)

    # Write the processed detections to per-camera JSON files in the specified output directory
    write_output_files(detections_by_cam_frame, Path(args.outdir), args.source, args.fps)


if __name__ == "__main__":
    main()
