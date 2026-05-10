from pathlib import Path
import yaml


def merge_yolo_datasets(yaml_paths: list[str], output_dir: str) -> Path:
    configs = []
    for p in yaml_paths:
        with open(p) as f:
            configs.append((Path(p).parent, yaml.safe_load(f)))

    all_names: list[str] = []
    seen: set[str] = set()
    for _, cfg in configs:
        for name in cfg.get("names", []):
            if name not in seen:
                all_names.append(name)
                seen.add(name)

    train_dirs, val_dirs = [], []
    for ds_root, cfg in configs:
        for key, bucket in (("train", train_dirs), ("val", val_dirs)):
            rel = cfg.get(key, f"images/{key}")
            if Path(rel).is_absolute():
                p = Path(rel)
            else:
                # Roboflow YAMLs emit e.g. '../train/images'; strip '..' to stay
                # inside the dataset folder instead of traversing up to the parent.
                clean = "/".join(part for part in Path(rel).parts if part != "..")
                p = ds_root / clean
            bucket.append(str(p))

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    merged_yaml = out / "data.yaml"
    with open(merged_yaml, "w") as f:
        yaml.dump({"train": train_dirs, "val": val_dirs, "nc": len(all_names), "names": all_names}, f)

    return merged_yaml
