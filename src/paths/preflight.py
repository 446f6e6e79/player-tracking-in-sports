from pathlib import Path
from collections.abc import Iterable


def preflight_output_paths(paths: Iterable[Path], force: bool) -> None:
    """Validate all output paths up front so we fail before expensive compute."""
    for path in paths:
        if path.exists() and not force:
            raise FileExistsError(
                f"Refusing to overwrite existing file: {path}. Pass --force to replace it."
            )
