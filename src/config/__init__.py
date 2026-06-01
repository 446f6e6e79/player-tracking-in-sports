"""Centralised config loader.

Usage:
    from src.config import get_config
    cfg = get_config()
    print(cfg.tracking.deep_sort.max_age)

The config is loaded from ``config.yaml`` at the repository root on the first
call to ``get_config()`` and cached for the remainder of the process.  Call
``load_config(path)`` explicitly to load from a different path or to reload.
"""

from pathlib import Path

import yaml

from src.config.schema import AppConfig

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_CONFIG_PATH = _REPO_ROOT / "config.yaml"

_config: AppConfig | None = None


def load_config(path: Path | None = None) -> AppConfig:
    """Load (or reload) the config from *path* (defaults to ``<repo>/config.yaml``)."""
    global _config
    p = path or _DEFAULT_CONFIG_PATH
    with open(p) as f:
        data = yaml.safe_load(f)
    _config = AppConfig.model_validate(data)
    return _config


def get_config() -> AppConfig:
    """Return the cached config, loading it on first access."""
    if _config is None:
        load_config()
    return _config  # type: ignore[return-value]
