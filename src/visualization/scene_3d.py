"""Animated 3D scene rendering of triangulated points using matplotlib.

Mirrors the minimap visual language (team colors, court lines) but in a true
3D axis so triangulated Z is visible.
"""
from __future__ import annotations

import numpy as np
import matplotlib

import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, FuncAnimation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  registers 3D projection

from src.geometry import court as cp
from src.types.geometry import FrameTriangulatedPoints, Point3D, TriangulationOutput


# World-frame axis bounds (meters) — court extents plus 1 m padding, Z up to 4 m.
_X_LIMITS_M = (-15.0, 15.0)
_Y_LIMITS_M = (-8.5, 8.5)
_Z_LIMITS_M = (0.0, 4.0)
_BALL_SIZE = 60
_PLAYER_SIZE = 90


def _bgr_to_rgb_unit(bgr: tuple[int, int, int]) -> tuple[float, float, float]:
    b, g, r = bgr
    return (r / 255.0, g / 255.0, b / 255.0)


def _draw_court_floor(ax: Axes3D) -> None:
    """Stroke the FIBA court polylines onto z=0 of the 3D axes."""
    for line in cp.court_lines():
        pts_m = line.pts / 1000.0
        xs = list(pts_m[:, 0])
        ys = list(pts_m[:, 1])
        zs = list(pts_m[:, 2])
        if line.closed:
            xs.append(xs[0])
            ys.append(ys[0])
            zs.append(zs[0])
        ax.plot(xs, ys, zs, color=(0.6, 0.6, 0.6), linewidth=0.8)
    for rim_x in (cp.LEFT_RIM_X, cp.RIGHT_RIM_X):
        ax.scatter([rim_x / 1000.0], [0.0], [3.05],
                   color=(1.0, 0.2, 0.2), s=20, depthshade=False)


def _split_points(points: list[Point3D]):
    """Group points by category for a single scatter call per category."""
    buckets: dict[str, list[Point3D]] = {"ball": [], "player": []}
    for p in points:
        buckets["ball" if p.is_ball else "player"].append(p)
    return buckets


def _scatter_frame(ax: Axes3D, frame: FrameTriangulatedPoints) -> None:
    buckets = _split_points(frame.points)
    for kind, pts in buckets.items():
        if not pts:
            continue
        xs = [p.x / 1000.0 for p in pts]
        ys = [p.y / 1000.0 for p in pts]
        zs = [p.z / 1000.0 for p in pts]
        colors = [_bgr_to_rgb_unit(p.color) for p in pts]
        size = _BALL_SIZE if kind == "ball" else _PLAYER_SIZE
        marker = "o" if kind == "ball" else "^"
        ax.scatter(xs, ys, zs, c=colors, s=size, marker=marker,
                   edgecolors="black", linewidths=0.5, depthshade=False)


def _configure_axes(ax: Axes3D) -> None:
    ax.set_xlim(*_X_LIMITS_M)
    ax.set_ylim(*_Y_LIMITS_M)
    ax.set_zlim(*_Z_LIMITS_M)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_box_aspect((
        _X_LIMITS_M[1] - _X_LIMITS_M[0],
        _Y_LIMITS_M[1] - _Y_LIMITS_M[0],
        _Z_LIMITS_M[1] - _Z_LIMITS_M[0],
    ))
    ax.view_init(elev=22, azim=-60)


def show_3d_scene_frames(
    triangulation: TriangulationOutput,
    *,
    n: int | None = None,
    indexes: list[int] | None = None,
) -> None:
    """Display a selection of frames from a TriangulationOutput as 3D subplots.
    Parameters:
        - triangulation: TriangulationOutput to inspect.
        - n: number of evenly-spaced frames to show.
        - indexes: explicit list of positions in triangulation.frames to show.
    """
    frames = triangulation.frames
    if indexes is not None:
        selected = [frames[i] for i in indexes]
    elif n is not None:
        idxs = np.linspace(0, len(frames) - 1, n, dtype=int).tolist()
        selected = [frames[i] for i in idxs]
    else:
        selected = [frames[0], frames[len(frames) // 2], frames[-1]]

    fig = plt.figure(figsize=(9 * len(selected), 6))
    for i, frame in enumerate(selected):
        ax = fig.add_subplot(1, len(selected), i + 1, projection="3d")
        _configure_axes(ax)
        _draw_court_floor(ax)
        _scatter_frame(ax, frame)
        ax.set_title(f"Frame {frame.frame_index}")
    plt.tight_layout()
    plt.show()


def produce_3d_scene_video(
    triangulation: TriangulationOutput,
    output_path: str,
    fps: float | None = None,
) -> None:
    """Render the triangulated scene as a 3D matplotlib animation MP4.

    Parameters:
        - triangulation: per-frame 3D points in world mm
        - output_path: destination MP4 path
        - fps: frame rate; falls back to triangulation.fps (or 25 if missing)
    """
    out_fps = fps if fps is not None else triangulation.fps
    if out_fps <= 0:
        out_fps = 25.0

    original_backend = matplotlib.get_backend()
    plt.switch_backend("Agg")

    fig = plt.figure(figsize=(10, 6))
    ax: Axes3D = fig.add_subplot(111, projection="3d")

    def render(frame_idx: int) -> None:
        ax.cla()
        _configure_axes(ax)
        _draw_court_floor(ax)
        frame = triangulation.frames[frame_idx]
        _scatter_frame(ax, frame)
        ax.set_title(f"frame {frame.frame_index}")

    anim = FuncAnimation(
        fig,
        render,
        frames=len(triangulation.frames),
        interval=1000.0 / out_fps,
        blit=False,
    )
    writer = FFMpegWriter(fps=int(out_fps))
    try:
        anim.save(output_path, writer=writer)
    finally:
        plt.close(fig)
        plt.switch_backend(original_backend)
    print(f"Video saved successfully at: {output_path}")
