import shutil
import yaml
from pathlib import Path

def merge_yolo_datasets(
    yaml_paths: list, 
    output_dir: str = "merged_dataset"
) -> Path:
    output_dir = Path(output_dir)
    datasets = []
    
    # Load all dataset configs and check for class consistency
    for i, p in enumerate(map(Path, yaml_paths)):
        with p.open() as f:
            config = yaml.safe_load(f)
        
        # Check that all datasets have the same class names (YOLO format)
        if datasets and datasets[0]['config'].get('names') != config.get('names'):
            raise ValueError(f"Class mismatch in {p}")
        
        # Store the dataset info for merging; we'll need the root to resolve relative paths
        datasets.append({
            'config': config,
            'root': p.parent,
            'id': i
        })

    # Use the class names from the first dataset
    names = datasets[0]['config'].get('names', {})
    splits = ['train', 'val', 'test']

    for split in splits:
        img_out = output_dir / "images" / split
        lbl_out = output_dir / "labels" / split
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        # If no dataset has this split, we can skip it (e.g. no test set)
        for dataset in datasets:
            if split not in dataset['config']:
                continue
            
            # Resolve paths relative to the dataset YAML location
            src_img_dir = (dataset['root'] / dataset['config'][split]).resolve()
            src_lbl_dir = src_img_dir.parents[1] / "labels" / src_img_dir.name

            # Copy images and labels, prefixing filenames with dataset ID to avoid collisions
            for img in src_img_dir.glob("*"):
                if not img.is_file(): continue
                
                # Prefix the image filename with the dataset ID to ensure uniqueness across datasets
                unique_name = f"ds{dataset['id']}_{img.name}"
                shutil.copy2(img, img_out / unique_name)

                # Copy the corresponding label file if it exists, using the same unique prefix
                lbl = src_lbl_dir / f"{img.stem}.txt"
                if lbl.exists():
                    shutil.copy2(lbl, lbl_out / f"{Path(unique_name).stem}.txt")

    # Final YAML configuration
    merged_config = {
        "path": str(output_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(names),
        "names": names
    }

    merged_yaml = output_dir / "data.yaml"
    with merged_yaml.open("w") as f:
        yaml.dump(merged_config, f, default_flow_style=False)

    return merged_yaml