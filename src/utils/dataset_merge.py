import yaml
from pathlib import Path


def _load_names(cfg: dict) -> list:
    names = cfg.get("names")
    if not names:
        raise ValueError("data.yaml is missing a non-empty `names` entry")
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names, key=int)]
    return list(names)


def _resolve_split_dir(cfg: dict, yaml_path: Path, key: str):
    split_val = cfg.get(key)
    if split_val is None:
        return None
    base = Path(cfg["path"]) if "path" in cfg else yaml_path.parent
    if not base.is_absolute():
        base = (yaml_path.parent / base).resolve()
    p = Path(split_val)
    return p.resolve() if p.is_absolute() else (base / p).resolve()


def merge_yolo_datasets(yaml_paths: list, output_yaml: str) -> Path:
    yaml_paths = [Path(p) for p in yaml_paths]
    if not yaml_paths:
        raise ValueError("yaml_paths is empty")

    configs = []
    for p in yaml_paths:
        with p.open() as f:
            cfg = yaml.safe_load(f)
        configs.append((p, cfg, _load_names(cfg)))

    names = configs[0][2]
    for p, _, n in configs[1:]:
        if n != names:
            raise ValueError(f"Class mismatch in {p}: {n} != {names}")

    split_lists = {"train": [], "val": [], "test": []}
    for p, cfg, _ in configs:
        for key in ("train", "test"):
            d = _resolve_split_dir(cfg, p, key)
            if d is not None:
                split_lists[key].append(str(d))
        d = _resolve_split_dir(cfg, p, "val") or _resolve_split_dir(cfg, p, "valid")
        if d is not None:
            split_lists["val"].append(str(d))

    output_yaml = Path(output_yaml)
    output_yaml.parent.mkdir(parents=True, exist_ok=True)

    merged = {
        "path": str(output_yaml.parent.resolve()),
        "nc": len(names),
        "names": names,
    }
    for key in ("train", "val", "test"):
        if split_lists[key]:
            merged[key] = split_lists[key]

    with output_yaml.open("w") as f:
        yaml.safe_dump(merged, f, default_flow_style=False, sort_keys=False)
    return output_yaml
