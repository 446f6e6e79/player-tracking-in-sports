"""Shared CLI argparse helpers used by the pipeline scripts."""
from __future__ import annotations

import argparse


def add_input_dir_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--input-dir", default=None,
        help="Directory containing the source videos (outN.mp4).",
    )


def add_output_dir_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--output-dir", default=None,
        help="Root directory for produced results.",
    )


def add_max_frames_arg(parser: argparse.ArgumentParser, help_text: str | None = None) -> None:
    parser.add_argument(
        "--max-frames", type=int, default=-1,
        help=help_text or "-1 to read every frame, otherwise stop after N frames.",
    )


def add_force_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--force", action="store_true",
        help="Overwrite existing pipeline outputs instead of failing.",
    )


def max_frames_or_none(value: int) -> int | None:
    """CLI sentinel -1 -> None; everything else passes through."""
    return None if value < 0 else value
