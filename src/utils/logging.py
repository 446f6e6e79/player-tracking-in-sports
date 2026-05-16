"""Logging utilities for the project.

Library modules call `get_logger(__name__)` at import time and never touch the
root logger or handlers. Entry points (CLI scripts, notebooks) call
`configure_logging()` exactly once to install the single shared handler.
"""
import logging
import sys


_LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
_DATE_FORMAT = "%H:%M:%S"
_NOISY_LIBS = ("ultralytics", "matplotlib", "PIL")


def get_logger(name: str) -> logging.Logger:
    """Return the module-level logger named `name` (typically `__name__`)."""
    return logging.getLogger(name)


def configure_logging(level: int = logging.INFO) -> None:
    """Attach a single stderr handler to the root logger.

    Idempotent: calling this more than once tunes the level but never stacks
    duplicate handlers (which would make every log line appear N times).
    Noisy third-party loggers are pinned to WARNING so they don't drown the
    pipeline output.
    """
    root = logging.getLogger()
    root.setLevel(level)

    handler = next(
        (h for h in root.handlers if getattr(h, "_project_handler", False)),
        None,
    )
    if handler is None:
        handler = logging.StreamHandler(stream=sys.stderr)
        handler._project_handler = True  # type: ignore[attr-defined]
        handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
        root.addHandler(handler)
    handler.setLevel(level)

    for noisy in _NOISY_LIBS:
        logging.getLogger(noisy).setLevel(logging.WARNING)
