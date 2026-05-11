import json
import os
from dataclasses import asdict, fields, is_dataclass
from pathlib import Path
from typing import Any, Self, get_args, get_origin, get_type_hints


class JsonSerializable:
    """
    Base class for dataclasses that can be serialized to and from JSON files.
    Subclasses must be decorated with @dataclass and contain only JSON-serializable fields (e.g. no custom objects or tuples)."""

    def write(self, path: str | Path, overwrite: bool = False) -> None:
        """Write this object to a JSON file at the given path. By default, this method will refuse to overwrite an existing file."""
        path = str(path)
        if not overwrite and os.path.exists(path):
            raise FileExistsError(
                f"Refusing to overwrite existing file: {path}. Pass overwrite=True to replace it."
            )
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f)

    @classmethod
    def read(cls, path: str | Path) -> Self:
        """Read an instance of this class from a JSON file at the given path."""
        path = str(path)
        if not os.path.exists(path):
            raise FileNotFoundError(f"{cls.__name__} file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return _from_dict(cls, data)


def _from_dict(cls: type, data: Any) -> Any:
    """Rebuild an instance of `cls` from a dict produced by `dataclasses.asdict`."""
    hints = get_type_hints(cls)
    kwargs = {}
    for f in fields(cls):
        kwargs[f.name] = _convert(hints[f.name], data[f.name])
    return cls(**kwargs)


def _convert(target_type: Any, value: Any) -> Any:
    """Recursively convert a value to the given target type, handling nested dataclasses and lists."""
    origin = get_origin(target_type)
    if origin is list:
        (item_type,) = get_args(target_type)
        return [_convert(item_type, v) for v in value]
    if is_dataclass(target_type):
        return _from_dict(target_type, value)
    return value
