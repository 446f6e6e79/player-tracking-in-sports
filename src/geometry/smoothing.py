import numpy as np

from src.types.geometry import FrameTriangulatedPoints, Point3D, TriangulationOutput

# Re-initiate a track's state after it goes missing for more than this many frames.
_MAX_GAP = 5


class KalmanFilter3D:
    """
    Forward-only constant-velocity Kalman filter for 3D world-frame points (mm).
    6D state: (x, y, z, vx, vy, vz). Observation: (x, y, z).

    Noise defaults (mm @ ~25 fps):
      std_pos=30  — triangulation jitter per frame;
      std_vel=80  — Δv per frame (~2 m/s player sprint);
      std_meas=50 — measurement uncertainty from triangulation.
    """

    def __init__(
        self,
        std_pos: float = 30.0,
        std_vel: float = 80.0,
        std_meas: float = 50.0,
    ) -> None:
        ndim, dt = 3, 1.0
        self._F = np.eye(2 * ndim)
        for i in range(ndim):
            self._F[i, ndim + i] = dt
        self._H = np.eye(ndim, 2 * ndim)
        self._Q = np.diag(np.array([std_pos**2] * ndim + [std_vel**2] * ndim))
        self._R = np.diag(np.array([std_meas**2] * ndim))
        self._P0_diag = np.array([std_pos**2] * ndim + [(2 * std_vel) ** 2] * ndim)

    def initiate(self, measurement: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mean = np.r_[np.asarray(measurement, dtype=float), np.zeros(3)]
        covariance = np.diag(self._P0_diag.copy())
        return mean, covariance

    def predict(
        self, mean: np.ndarray, covariance: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        new_mean = self._F @ mean
        new_cov = self._F @ covariance @ self._F.T + self._Q
        return new_mean, new_cov

    def update(
        self,
        mean: np.ndarray,
        covariance: np.ndarray,
        measurement: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        proj_mean = self._H @ mean
        proj_cov = self._H @ covariance @ self._H.T + self._R

        chol = np.linalg.cholesky(proj_cov)
        kalman_gain = np.linalg.solve(
            chol.T, np.linalg.solve(chol, (covariance @ self._H.T).T)
        ).T

        residual = np.asarray(measurement, dtype=float) - proj_mean
        new_mean = mean + kalman_gain @ residual
        new_cov = covariance - kalman_gain @ proj_cov @ kalman_gain.T
        return new_mean, new_cov


def smooth_triangulation(
    triangulation: TriangulationOutput,
    *,
    std_pos: float = 30.0,
    std_vel: float = 60.0,
    std_meas: float = 80.0,
    max_gap: int = _MAX_GAP,
    fill_gaps: bool = True,
) -> TriangulationOutput:
    """
    Apply a per-class_id forward Kalman filter to every Point3D in `triangulation`.

    Frames where a class_id is absent can be gap-filled (prediction only) to
    keep dots visible. After a gap of more than `max_gap` frames the state is
    dropped and re-initiated on the next observation to avoid stale velocity
    poisoning. The frame structure is preserved unless `fill_gaps` is True.
    """
    kf = KalmanFilter3D(std_pos=std_pos, std_vel=std_vel, std_meas=std_meas)
    states: dict[int, tuple[np.ndarray, np.ndarray, str]] = {}
    last_seen: dict[int, int] = {}

    new_frames: list[FrameTriangulatedPoints] = []
    for frame in sorted(triangulation.frames, key=lambda f: f.frame_index):
        fi = frame.frame_index
        smoothed_points: list[Point3D] = []
        observed_ids: set[int] = set()
        for p in frame.points:
            cid = p.class_id
            meas = np.array([p.x, p.y, p.z], dtype=float)
            observed_ids.add(cid)

            gap = fi - last_seen[cid] if cid in last_seen else max_gap + 1
            if cid not in states or gap > max_gap:
                mean, cov = kf.initiate(meas)
                sm = meas
            else:
                mean, cov = kf.predict(*states[cid][:2])
                mean, cov = kf.update(mean, cov, meas)
                sm = mean[:3]

            states[cid] = (mean, cov, p.class_name)
            last_seen[cid] = fi
            smoothed_points.append(
                Point3D(
                    x=float(sm[0]),
                    y=float(sm[1]),
                    z=float(sm[2]),
                    class_id=p.class_id,
                    class_name=p.class_name,
                )
            )

        if fill_gaps:
            expired: list[int] = []
            for cid, (mean, cov, class_name) in list(states.items()):
                if cid in observed_ids:
                    continue
                gap = fi - last_seen.get(cid, fi)
                if gap > max_gap:
                    expired.append(cid)
                    continue
                mean, cov = kf.predict(mean, cov)
                states[cid] = (mean, cov, class_name)
                smoothed_points.append(
                    Point3D(
                        x=float(mean[0]),
                        y=float(mean[1]),
                        z=float(mean[2]),
                        class_id=cid,
                        class_name=class_name,
                    )
                )

            for cid in expired:
                states.pop(cid, None)
                last_seen.pop(cid, None)

        new_frames.append(
            FrameTriangulatedPoints(frame_index=fi, points=smoothed_points)
        )

    return TriangulationOutput(
        fps=triangulation.fps,
        camera_ids=triangulation.camera_ids,
        frames=new_frames,
    )
