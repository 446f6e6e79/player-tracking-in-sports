"""Constant-velocity Kalman filter used by the DeepSORT-style tracker.

State (8D):    x = [cx, cy, a, h, vx, vy, va, vh]
                       └─ position ──┘ └─ velocity ─┘
    cx, cy : bbox center pixel coords
    a      : aspect ratio = w / h
    h      : bbox height in pixels

Measurement (4D): z = [cx, cy, a, h]   (we observe positions, not velocities)

Motion model is constant velocity (positions += velocities * dt).
Process and measurement noise std-devs are scaled with the bbox height h:
larger boxes are typically closer/bigger on screen, so absolute pixel
uncertainty grows with h. This is the standard DeepSORT recipe and lets
one filter cover players (tall) and the ball (tiny) without retuning.
"""
import numpy as np


class KalmanFilter:
    def __init__(self) -> None:
        ndim, dt = 4, 1.0

        # F: state transition. Position[i] gets velocity[i] * dt added each step,
        # velocities are unchanged. Identity-with-velocity-coupling.
        self._F = np.eye(2 * ndim)
        for i in range(ndim):
            self._F[i, ndim + i] = dt

        # H: measurement matrix. We observe the first 4 components (positions).
        self._H = np.eye(ndim, 2 * ndim)

        # Noise scaling weights — std proportional to bbox height.
        self._std_weight_position = 1.0 / 20
        self._std_weight_velocity = 1.0 / 160

    def initiate(self, measurement: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Build (mean, covariance) from the very first detection.

        Velocities start at 0 (we have no motion evidence yet) but with wide
        variance so the first few updates can correct them quickly.
        """
        mean_pos = np.asarray(measurement, dtype=float)
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]

        h = mean_pos[3]
        std = [
            2 * self._std_weight_position * h,    # cx
            2 * self._std_weight_position * h,    # cy
            1e-2,                                 # a (aspect ratio is unitless)
            2 * self._std_weight_position * h,    # h
            10 * self._std_weight_velocity * h,   # vx (unknown -> wide)
            10 * self._std_weight_velocity * h,   # vy
            1e-5,                                 # va
            10 * self._std_weight_velocity * h,   # vh
        ]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(
        self, mean: np.ndarray, covariance: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Advance the state one frame forward.

            x' = F x
            P' = F P F^T + Q

        Q (process noise) grows with the current bbox height.
        """
        h = mean[3]
        std_pos = [
            self._std_weight_position * h,
            self._std_weight_position * h,
            1e-2,
            self._std_weight_position * h,
        ]
        std_vel = [
            self._std_weight_velocity * h,
            self._std_weight_velocity * h,
            1e-5,
            self._std_weight_velocity * h,
        ]
        Q = np.diag(np.square(np.r_[std_pos, std_vel]))

        new_mean = self._F @ mean
        new_cov = self._F @ covariance @ self._F.T + Q
        return new_mean, new_cov

    def project(
        self, mean: np.ndarray, covariance: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Project state into measurement space (and add measurement noise R).

        Returns the predicted measurement and its innovation covariance S.
        """
        h = mean[3]
        std = [
            self._std_weight_position * h,
            self._std_weight_position * h,
            1e-1,
            self._std_weight_position * h,
        ]
        R = np.diag(np.square(std))

        proj_mean = self._H @ mean
        proj_cov = self._H @ covariance @ self._H.T + R
        return proj_mean, proj_cov

    def update(
        self,
        mean: np.ndarray,
        covariance: np.ndarray,
        measurement: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Standard Kalman update.

            y = z - H x         (innovation: measurement - prediction)
            S = H P H^T + R     (innovation covariance, from project())
            K = P H^T S^-1      (Kalman gain)
            x_new = x + K y
            P_new = (I - K H) P

        We solve K via Cholesky factor of S instead of inverting S — same
        result, better numerical stability.
        """
        proj_mean, proj_cov = self.project(mean, covariance)

        # Solve K^T from S K^T = (P H^T)^T using Cholesky-factored S.
        chol = np.linalg.cholesky(proj_cov)
        kalman_gain = np.linalg.solve(
            chol.T, np.linalg.solve(chol, (covariance @ self._H.T).T)
        ).T

        innovation = np.asarray(measurement, dtype=float) - proj_mean
        new_mean = mean + kalman_gain @ innovation
        new_cov = covariance - kalman_gain @ proj_cov @ kalman_gain.T
        return new_mean, new_cov
