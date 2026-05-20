import numpy as np
import dask.array as da
from scipy import linalg

from tfilterspy.base_estimator import BaseEstimator


class KalmanFilter(BaseEstimator):
    r"""
    Kalman Filter for linear Gaussian state-space models.

    Efficient numpy core with optional Dask support for batch processing.
    Implements the full sklearn-compatible API: fit / predict / score.

    State-space model::

        x_{k+1} = F @ x_k + w_k,    w_k ~ N(0, Q)
        z_k     = H @ x_k + v_k,     v_k ~ N(0, R)

    Parameters
    ----------
    F : ndarray, shape (n_state, n_state)
        State transition matrix.
    H : ndarray, shape (n_obs, n_state)
        Observation matrix mapping state to measurement space.
    Q : ndarray, shape (n_state, n_state)
        Process noise covariance.
    R : ndarray, shape (n_obs, n_obs)
        Observation noise covariance.
    x0 : ndarray, shape (n_state,)
        Initial state estimate.
    P0 : ndarray, shape (n_state, n_state)
        Initial state covariance.
    store_covariances : bool, optional
        If True (default), stores full covariance history. Set to False
        to save memory on very long time series (disables smooth()).

    Examples
    --------
    >>> import numpy as np
    >>> from tfilterspy import KalmanFilter
    >>> F = np.eye(2); H = np.eye(2)
    >>> Q = np.eye(2) * 0.1; R = np.eye(2) * 0.5
    >>> kf = KalmanFilter(F, H, Q, R, np.zeros(2), np.eye(2))
    >>> kf.fit(np.random.randn(100, 2))
    >>> states = kf.predict()
    >>> smoothed, _ = kf.smooth()
    """

    def __init__(self, F, H, Q, R, x0, P0, store_covariances=True):
        super().__init__()
        F, H, Q, R, x0, P0 = (np.asarray(a, dtype=np.float64) for a in [F, H, Q, R, x0, P0])

        n_state = F.shape[0]
        if F.shape != (n_state, n_state):
            raise ValueError("State transition matrix (F) must be square.")
        if H.shape[1] != n_state:
            raise ValueError("Observation matrix (H) columns must match state dimension.")
        if Q.shape != (n_state, n_state):
            raise ValueError("Process noise covariance (Q) dimensions must match F.")
        if R.shape[0] != R.shape[1] or R.shape[0] != H.shape[0]:
            raise ValueError("Observation noise covariance (R) must be square with dimension matching H rows.")
        if x0.shape != (n_state,):
            raise ValueError("Initial state (x0) must have length n_state.")
        if P0.shape != (n_state, n_state):
            raise ValueError("Initial covariance (P0) dimensions must match F.")

        self.F = F
        self.H = H
        self.Q = Q
        self.R = R
        self.x0 = x0
        self.P0 = P0
        self.n_state = n_state
        self.n_obs = H.shape[0]
        self.store_covariances = store_covariances
        self.is_fitted_ = False

    def fit(self, X):
        """
        Run the Kalman filter forward pass on measurements.

        Parameters
        ----------
        X : ndarray or dask.array, shape (n_timesteps, n_obs)
            Measurement sequence. 1-D arrays are reshaped to (n, 1).

        Returns
        -------
        self
        """
        if isinstance(X, da.Array):
            X = X.compute()
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if X.shape[1] != self.n_obs:
            raise ValueError(
                f"Expected {self.n_obs} observation dimensions, got {X.shape[1]}."
            )

        self.measurements_ = X
        self._forward_pass()
        return self

    def _forward_pass(self):
        n = len(self.measurements_)
        ns = self.n_state
        no = self.n_obs
        F, H, Q, R = self.F, self.H, self.Q, self.R
        I_ns = np.eye(ns)

        self.filtered_states_ = np.empty((n, ns))
        self.predicted_states_ = np.empty((n, ns))
        self.innovations_ = np.empty((n, no))
        self.log_likelihood_ = 0.0

        if self.store_covariances:
            self.filtered_covs_ = np.empty((n, ns, ns))
            self.predicted_covs_ = np.empty((n, ns, ns))
            self.innovation_covs_ = np.empty((n, no, no))

        x = self.x0.copy()
        P = self.P0.copy()

        for i in range(n):
            # --- Predict ---
            x_pred = F @ x
            P_pred = F @ P @ F.T + Q

            self.predicted_states_[i] = x_pred
            if self.store_covariances:
                self.predicted_covs_[i] = P_pred

            # --- Innovation ---
            z = self.measurements_[i]
            y = z - H @ x_pred
            S = H @ P_pred @ H.T + R

            self.innovations_[i] = y
            if self.store_covariances:
                self.innovation_covs_[i] = S

            # --- Update (Joseph form for numerical stability) ---
            S_inv = linalg.solve(S, np.eye(no), assume_a="pos")
            K = P_pred @ H.T @ S_inv

            x = x_pred + K @ y
            IKH = I_ns - K @ H
            P = IKH @ P_pred @ IKH.T + K @ R @ K.T

            self.filtered_states_[i] = x
            if self.store_covariances:
                self.filtered_covs_[i] = P

            # --- Log-likelihood ---
            sign, logdet = np.linalg.slogdet(S)
            if sign > 0:
                self.log_likelihood_ += -0.5 * (
                    logdet + y @ S_inv @ y + no * np.log(2 * np.pi)
                )

        self.is_fitted_ = True

    def predict(self):
        """
        Return filtered state estimates.

        Returns
        -------
        filtered_states : ndarray, shape (n_timesteps, n_state)
        """
        self._check_fitted()
        return self.filtered_states_

    def smooth(self):
        """
        Rauch-Tung-Striebel (RTS) backward smoothing pass.

        Requires ``store_covariances=True`` (the default).

        Returns
        -------
        smoothed_states : ndarray, shape (n_timesteps, n_state)
        smoothed_covs : ndarray, shape (n_timesteps, n_state, n_state)
        """
        self._check_fitted()
        if not self.store_covariances:
            raise RuntimeError("smooth() requires store_covariances=True.")

        n = len(self.filtered_states_)
        ns = self.n_state

        smoothed_states = np.empty((n, ns))
        smoothed_covs = np.empty((n, ns, ns))
        smoothed_states[-1] = self.filtered_states_[-1]
        smoothed_covs[-1] = self.filtered_covs_[-1]

        for i in range(n - 2, -1, -1):
            P_pred_inv = linalg.inv(self.predicted_covs_[i + 1])
            C = self.filtered_covs_[i] @ self.F.T @ P_pred_inv
            smoothed_states[i] = (
                self.filtered_states_[i]
                + C @ (smoothed_states[i + 1] - self.predicted_states_[i + 1])
            )
            smoothed_covs[i] = (
                self.filtered_covs_[i]
                + C @ (smoothed_covs[i + 1] - self.predicted_covs_[i + 1]) @ C.T
            )

        return smoothed_states, smoothed_covs

    def filter_step(self, z):
        """
        Single-step online filtering.

        Parameters
        ----------
        z : ndarray, shape (n_obs,)
            Single measurement vector.

        Returns
        -------
        x : ndarray, shape (n_state,)
            Updated state estimate.
        P : ndarray, shape (n_state, n_state)
            Updated covariance.
        """
        if not hasattr(self, "_online_x"):
            self._online_x = self.x0.copy()
            self._online_P = self.P0.copy()

        z = np.asarray(z, dtype=np.float64)
        F, H, Q, R = self.F, self.H, self.Q, self.R

        x_pred = F @ self._online_x
        P_pred = F @ self._online_P @ F.T + Q

        y = z - H @ x_pred
        S = H @ P_pred @ H.T + R
        K = P_pred @ H.T @ linalg.inv(S)

        self._online_x = x_pred + K @ y
        IKH = np.eye(self.n_state) - K @ H
        self._online_P = IKH @ P_pred @ IKH.T + K @ R @ K.T

        return self._online_x.copy(), self._online_P.copy()

    def forecast(self, n_steps):
        """
        Forecast future states beyond the observed data.

        Parameters
        ----------
        n_steps : int
            Number of steps to forecast.

        Returns
        -------
        forecasted_states : ndarray, shape (n_steps, n_state)
        forecasted_covs : ndarray, shape (n_steps, n_state, n_state)
        """
        self._check_fitted()
        x = self.filtered_states_[-1].copy()
        P = self.filtered_covs_[-1].copy() if self.store_covariances else self.P0.copy()

        forecasted_states = np.empty((n_steps, self.n_state))
        forecasted_covs = np.empty((n_steps, self.n_state, self.n_state))

        for i in range(n_steps):
            x = self.F @ x
            P = self.F @ P @ self.F.T + self.Q
            forecasted_states[i] = x
            forecasted_covs[i] = P

        return forecasted_states, forecasted_covs

    def run_filter(self, measurements):
        """Legacy method. Use fit() + predict() instead."""
        self.fit(measurements)
        return self.filtered_states_, self.innovations_


class DaskKalmanFilter(KalmanFilter):
    """
    Backward-compatible Kalman Filter wrapper.

    Accepts the same constructor signature as the original DaskKalmanFilter
    and delegates to the new efficient KalmanFilter implementation.
    """

    def __init__(
        self,
        state_transition_matrix,
        observation_matrix,
        process_noise_cov,
        observation_noise_cov,
        initial_state,
        initial_covariance,
        chunk_size=64,
        estimation_strategy="residual_analysis",
    ):
        super().__init__(
            F=state_transition_matrix,
            H=observation_matrix,
            Q=process_noise_cov,
            R=observation_noise_cov,
            x0=initial_state,
            P0=initial_covariance,
        )
        self.chunk_size = chunk_size
        self.estimation_strategy = estimation_strategy

    def predict(self):
        """Return filtered states as a Dask array for backward compatibility."""
        self._check_fitted()
        return da.from_array(self.filtered_states_, chunks="auto")
