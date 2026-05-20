import numpy as np
from scipy import linalg

from tfilterspy.base_estimator import BaseEstimator


class ExtendedKalmanFilter(BaseEstimator):
    r"""
    Extended Kalman Filter for nonlinear state-space models.

    Linearizes the dynamics using first-order Taylor expansion (Jacobians)
    around the current state estimate at each time step.

    State-space model::

        x_{k+1} = f(x_k) + w_k,    w_k ~ N(0, Q)
        z_k     = h(x_k) + v_k,     v_k ~ N(0, R)

    Parameters
    ----------
    f : callable
        State transition function ``f(x) -> x_next``.
    h : callable
        Observation function ``h(x) -> z``.
    F_jacobian : callable
        Jacobian of f: ``F_jacobian(x) -> ndarray (n_state, n_state)``.
    H_jacobian : callable
        Jacobian of h: ``H_jacobian(x) -> ndarray (n_obs, n_state)``.
    Q : ndarray, shape (n_state, n_state)
        Process noise covariance.
    R : ndarray, shape (n_obs, n_obs)
        Observation noise covariance.
    x0 : ndarray, shape (n_state,)
        Initial state estimate.
    P0 : ndarray, shape (n_state, n_state)
        Initial state covariance.

    Examples
    --------
    >>> import numpy as np
    >>> from tfilterspy import ExtendedKalmanFilter
    >>> def f(x): return x  # identity dynamics
    >>> def h(x): return x[:1]  # observe first state only
    >>> def Fj(x): return np.eye(2)
    >>> def Hj(x): return np.array([[1.0, 0.0]])
    >>> ekf = ExtendedKalmanFilter(f, h, Fj, Hj,
    ...     Q=np.eye(2)*0.1, R=np.eye(1)*0.5,
    ...     x0=np.zeros(2), P0=np.eye(2))
    >>> ekf.fit(np.random.randn(50, 1))
    >>> states = ekf.predict()
    """

    def __init__(self, f, h, F_jacobian, H_jacobian, Q, R, x0, P0):
        super().__init__()
        self.f = f
        self.h = h
        self.F_jacobian = F_jacobian
        self.H_jacobian = H_jacobian
        self.Q = np.asarray(Q, dtype=np.float64)
        self.R = np.asarray(R, dtype=np.float64)
        self.x0 = np.asarray(x0, dtype=np.float64)
        self.P0 = np.asarray(P0, dtype=np.float64)
        self.n_state = len(x0)
        self.n_obs = self.R.shape[0]
        self.is_fitted_ = False

    def fit(self, X):
        """
        Run the EKF forward pass on measurements.

        Parameters
        ----------
        X : ndarray, shape (n_timesteps, n_obs)

        Returns
        -------
        self
        """
        if hasattr(X, "compute"):
            X = X.compute()
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        self.measurements_ = X
        n = len(X)
        ns = self.n_state
        no = self.n_obs
        I_ns = np.eye(ns)

        self.filtered_states_ = np.empty((n, ns))
        self.filtered_covs_ = np.empty((n, ns, ns))
        self.predicted_states_ = np.empty((n, ns))
        self.predicted_covs_ = np.empty((n, ns, ns))
        self.log_likelihood_ = 0.0

        x = self.x0.copy()
        P = self.P0.copy()

        for i in range(n):
            F = self.F_jacobian(x)

            x_pred = self.f(x)
            P_pred = F @ P @ F.T + self.Q

            self.predicted_states_[i] = x_pred
            self.predicted_covs_[i] = P_pred

            H = self.H_jacobian(x_pred)
            z = X[i]
            y = z - self.h(x_pred)
            S = H @ P_pred @ H.T + self.R

            S_inv = linalg.solve(S, np.eye(no), assume_a="pos")
            K = P_pred @ H.T @ S_inv

            x = x_pred + K @ y
            IKH = I_ns - K @ H
            P = IKH @ P_pred @ IKH.T + K @ self.R @ K.T

            self.filtered_states_[i] = x
            self.filtered_covs_[i] = P

            sign, logdet = np.linalg.slogdet(S)
            if sign > 0:
                self.log_likelihood_ += -0.5 * (
                    logdet + y @ S_inv @ y + no * np.log(2 * np.pi)
                )

        self.is_fitted_ = True
        return self

    def predict(self):
        """Return filtered state estimates."""
        self._check_fitted()
        return self.filtered_states_

    def smooth(self):
        """
        RTS smoother for EKF (re-linearizes at filtered states).

        Returns
        -------
        smoothed_states : ndarray, shape (n_timesteps, n_state)
        smoothed_covs : ndarray, shape (n_timesteps, n_state, n_state)
        """
        self._check_fitted()
        n = len(self.filtered_states_)
        ns = self.n_state

        smoothed_states = np.empty((n, ns))
        smoothed_covs = np.empty((n, ns, ns))
        smoothed_states[-1] = self.filtered_states_[-1]
        smoothed_covs[-1] = self.filtered_covs_[-1]

        for i in range(n - 2, -1, -1):
            F = self.F_jacobian(self.filtered_states_[i])
            C = self.filtered_covs_[i] @ F.T @ linalg.inv(self.predicted_covs_[i + 1])
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
        """Single-step online EKF update."""
        if not hasattr(self, "_online_x"):
            self._online_x = self.x0.copy()
            self._online_P = self.P0.copy()

        z = np.asarray(z, dtype=np.float64)

        F = self.F_jacobian(self._online_x)
        x_pred = self.f(self._online_x)
        P_pred = F @ self._online_P @ F.T + self.Q

        H = self.H_jacobian(x_pred)
        y = z - self.h(x_pred)
        S = H @ P_pred @ H.T + self.R
        K = P_pred @ H.T @ linalg.inv(S)

        self._online_x = x_pred + K @ y
        IKH = np.eye(self.n_state) - K @ H
        self._online_P = IKH @ P_pred @ IKH.T + K @ self.R @ K.T

        return self._online_x.copy(), self._online_P.copy()
