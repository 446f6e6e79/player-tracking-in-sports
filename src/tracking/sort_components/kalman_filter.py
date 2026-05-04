import numpy as np


class KalmanFilter:
    """
    Kalman filter for tracking bounding boxes in image space.
    The 8D state space is (cx, cy, a, h, vx, vy, va, vh), where:
      - cx, cy: center of the bounding box in pixel coordinates.
      - a: aspect ratio (width / height) of the bounding box.
      - h: height of the bounding box in pixels.
      - vx, vy: velocity of the center in pixels/frame.
      - va: velocity of the aspect ratio (unitless).
      - vh: velocity of the height in pixels/frame.
    """
    def __init__(self) -> None:
        # Number of dimensions in the position part of the state (cx, cy, a, h).
        ndim = 4
        # We work in pixel space, so dt=1 (1 frame per time step)
        dt = 1.0

        # F: state transition. Position[i] gets velocity[i] * dt added each step,
        # velocities are unchanged. Identity-with-velocity-coupling.
        self._F = np.eye(2 * ndim)
        for i in range(ndim):
            self._F[i, ndim + i] = dt

        # H: measurement matrix. We observe the first 4 components (positions).
        # It maps the 8D state space to the 4D measurement space.
        # Each row corresponds to a measurement dimension, each column to a state dimension.
        self._H = np.eye(ndim, 2 * ndim)

        # Noise scaling weights — std proportional to bbox height.
        self._std_weight_position = 1.0 / 20
        self._std_weight_velocity = 1.0 / 160

    def initiate(
        self, 
        measurement: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Create track from unassociated measurement.
        Parameters:
            - measurement: The observed bounding box in the format [cx, cy, a, h].
        Returns:
            - mean: The initial state mean vector (8D).
            - covariance: The initial state covariance matrix (8x8).
        The initial mean has the observed position and zero velocity.
        
        The initial covariance is diagonal, with std-devs scaled by the bbox height.
        This is done because larger boxes (closer objects) typically have higher absolute pixel uncertainty.
        The velocity components are initialized with a higher uncertainty since we have no prior on the object's motion.
        """
        # Initialize mean state: observed position + zero velocity.
        mean_pos = np.asarray(measurement, dtype=float)
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]

        # Get the bbox height to scale the initial uncertainty.
        h = mean_pos[3]
        std = [
            2 * self._std_weight_position * h,    # cx
            2 * self._std_weight_position * h,    # cy
            1e-2,                                 # a (aspect ratio is unitless)
            2 * self._std_weight_position * h,    # h
            #Velocies are more uncertain, so we use a higher multiplier.
            10 * self._std_weight_velocity * h,   # vx
            10 * self._std_weight_velocity * h,   # vy
            1e-5,                                 # va
            10 * self._std_weight_velocity * h,   # vh
        ]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(
        self, 
        mean: np.ndarray, 
        covariance: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Run Kalman filter prediction step.
        Given the current mean and covariance, apply the motion model to predict 
        the next state.
        """
        h = mean[3]
        # Define the process noise covariance Q, which models the uncertainty in the motion.
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

        # Update the mean and covariance using the state transition matrix F and process noise Q.
        new_mean = self._F @ mean
        new_cov = self._F @ covariance @ self._F.T + Q
        return new_mean, new_cov

    def project(
        self, 
        mean: np.ndarray, 
        covariance: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Project state into measurement space (and add measurement noise R).
        Returns the predicted measurement and its innovation covariance S.
        """
        h = mean[3]
        # Define the measurement noise covariance R, which models the uncertainty in the measurements.
        std = [
            self._std_weight_position * h,
            self._std_weight_position * h,
            1e-1,
            self._std_weight_position * h,
        ]
        R = np.diag(np.square(std))

        # Project the mean and covariance into measurement space using the measurement matrix H.
        proj_mean = self._H @ mean
        proj_cov = self._H @ covariance @ self._H.T + R
        return proj_mean, proj_cov

    def update(
        self,
        mean: np.ndarray,
        covariance: np.ndarray,
        measurement: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Standard Kalman update. Given a predicted mean and covariance,
        and a new measurement, compute the Kalman gain and update the state.
        The equations are:
            y = z - H x         (innovation: measurement - prediction)
            S = H P H^T + R     (innovation covariance, from project())
            K = P H^T S^-1      (Kalman gain)
            x_new = x + K y
            P_new = (I - K H) P
        We solve K via Cholesky factor of S instead of inverting S — same
        result, better numerical stability.

        Parameters:
            - mean: The predicted state mean (8D).
            - covariance: The predicted state covariance (8x8).
            - measurement: The new measurement (4D).
        Returns:
            - new_mean: The updated state mean after incorporating the measurement.
            - new_cov: The updated state covariance after incorporating the measurement.
        """
        # Map the predicted mean and covariance into measurement space to get the expected measurement and its covariance.
        proj_mean, proj_cov = self.project(mean, covariance)

        # Solve K^T from S K^T = (P H^T)^T using Cholesky-factored S.
        chol = np.linalg.cholesky(proj_cov)
        kalman_gain = np.linalg.solve(
            chol.T, np.linalg.solve(chol, (covariance @ self._H.T).T)
        ).T
        # Compute the measurement residual (the difference between the actual measurement and the predicted measurement).
        measurement_residual = np.asarray(measurement, dtype=float) - proj_mean
        
        # Update the mean and covariance using the Kalman gain and the innovation.
        new_mean = mean + kalman_gain @ measurement_residual
        new_cov = covariance - kalman_gain @ proj_cov @ kalman_gain.T
        return new_mean, new_cov
