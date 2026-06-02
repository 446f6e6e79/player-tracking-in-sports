from __future__ import annotations
import logging
import numpy as np

from src.config import get_config
from src.types.geometry import (
    FrameTriangulatedPoints,
    Point3D,
    TriangulationOutput,
)

logger = logging.getLogger(__name__)


def _transition() -> np.ndarray:
    f = np.eye(6)
    for i in range(3):
        f[i, i + 3] = 1.0
    return f


def _process_noise(std_pos: float, std_vel: float) -> np.ndarray:
    q = np.zeros((6, 6))
    for i in range(3):
        q[i, i] = std_pos**2
        q[i + 3, i + 3] = std_vel**2
    return q


def _measurement_matrix() -> np.ndarray:
    h = np.zeros((3, 6))
    h[0, 0] = h[1, 1] = h[2, 2] = 1.0
    return h


def _smooth_segment(
    measurements: list[np.ndarray | None],
    std_pos: float,
    std_vel: float,
    std_meas: float,
    gate_chi2: float,
) -> list[np.ndarray]:
    """Forward Kalman + RTS backward pass over one contiguous segment.

    ``measurements`` is one entry per consecutive frame (``None`` = gap-filled).
    The first entry is always a real measurement. Returns the smoothed position
    (length-3) for every frame in the segment.
    """
    f = _transition()
    h = _measurement_matrix()
    q = _process_noise(std_pos, std_vel)
    r = np.eye(3) * (std_meas**2)

    n = len(measurements)
    x_prior = [np.zeros(6) for _ in range(n)]
    p_prior = [np.eye(6) for _ in range(n)]
    x_post = [np.zeros(6) for _ in range(n)]
    p_post = [np.eye(6) for _ in range(n)]

    x_post[0][:3] = measurements[0]
    p_post[0] = np.eye(6) * 1000.0
    x_prior[0] = x_post[0].copy()
    p_prior[0] = p_post[0].copy()

    for i in range(1, n):
        x_prior[i] = f @ x_post[i - 1]
        p_prior[i] = f @ p_post[i - 1] @ f.T + q
        z = measurements[i]
        accept = z is not None
        if accept:
            y = z - h @ x_prior[i]
            s = h @ p_prior[i] @ h.T + r
            s_inv = np.linalg.inv(s)
            if float(y @ s_inv @ y) > gate_chi2:
                accept = False
        if accept:
            k = p_prior[i] @ h.T @ s_inv
            x_post[i] = x_prior[i] + k @ y
            p_post[i] = (np.eye(6) - k @ h) @ p_prior[i]
        else:
            x_post[i] = x_prior[i]
            p_post[i] = p_prior[i]

    x_smooth = [x.copy() for x in x_post]
    for i in range(n - 2, -1, -1):
        c = p_post[i] @ f.T @ np.linalg.inv(p_prior[i + 1])
        x_smooth[i] = x_post[i] + c @ (x_smooth[i + 1] - x_prior[i + 1])

    return [x[:3].copy() for x in x_smooth]


def smooth_triangulation(
    triangulation: TriangulationOutput,
    std_pos: float,
    std_vel: float,
    std_meas: float,
) -> TriangulationOutput:
    """Apply per-class RTS smoothing, splitting on gaps longer than ``geometry.smoothing.max_gap``."""
    smooth_cfg = get_config().geometry.smoothing
    ball_class_name = get_config().classes.ball_class_name

    # Gather each class's observations in frame order.
    per_class: dict[int, list[tuple[int, np.ndarray]]] = {}
    class_names: dict[int, str] = {}
    for frame in triangulation.frames:
        for p in frame.points:
            class_names[p.class_id] = p.class_name
            per_class.setdefault(p.class_id, []).append(
                (frame.frame_index, np.array([p.x, p.y, p.z], dtype=np.float64))
            )

    # frame_index -> list of smoothed Point3D
    out_points: dict[int, list[Point3D]] = {f.frame_index: [] for f in triangulation.frames}

    for class_id, observations in per_class.items():
        observations.sort(key=lambda o: o[0])
        name = class_names[class_id]
        is_ball = name == ball_class_name

        # Split into segments wherever the gap exceeds the configured max_gap.
        segments: list[list[tuple[int, np.ndarray]]] = []
        current = [observations[0]]
        for prev, nxt in zip(observations, observations[1:]):
            if nxt[0] - prev[0] > smooth_cfg.max_gap:
                segments.append(current)
                current = [nxt]
            else:
                current.append(nxt)
        segments.append(current)

        for segment in segments:
            start = segment[0][0]
            end = segment[-1][0]
            by_frame = {idx: pos for idx, pos in segment}
            grid = list(range(start, end + 1))
            measurements = [by_frame.get(idx) for idx in grid]
            smoothed = _smooth_segment(
                measurements, std_pos, std_vel, std_meas, smooth_cfg.gate_chi2
            )
            for idx, pos in zip(grid, smoothed):
                if idx not in out_points:
                    continue
                z = 0.0 if not is_ball else float(pos[2])
                out_points[idx].append(
                    Point3D(
                        x=float(pos[0]),
                        y=float(pos[1]),
                        z=z,
                        class_id=class_id,
                        class_name=name,
                    )
                )

    smoothed_frames = [
        FrameTriangulatedPoints(frame_index=f.frame_index, points=out_points[f.frame_index])
        for f in triangulation.frames
    ]
    return TriangulationOutput(
        frames=smoothed_frames,
        fps=triangulation.fps,
        camera_ids=triangulation.camera_ids,
    )
