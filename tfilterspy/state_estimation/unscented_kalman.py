import numpy as np
from scipy import linalg

from tfilterspy.base_estimator import BaseEstimator


class UnscentedKalmanFilter(BaseEstimator):
    r"""
    Unscented Kalman Filter for nonlinear state-space models.

    Uses the unscented transform with sigma points to propagate mean and
    covariance through nonlinear functions — no Jacobians required.

    State-space model::

        x_{k+1} = f(x_k) + w_k,    w_k ~ N(0, Q)
        z_k     = h(x_k) + v_k,     v_k ~ N(0, R)

    Parameters
    ----------
    f : callable
        State transition function ``f(x) -> x_next``.
    h : callable
        Observation function ``h(x) -> z``.
    Q : ndarray, shape (n_state, n_state)
        Process noise covariance.
    R : ndarray, shape (n_obs, n_obs)
        Observation noise covariance.
    x0 : ndarray, shape (n_state,)
        Initial state estimate.
    P0 : ndarray, shape (n_state, n_state)
        Initial state covariance.
    alpha : float, optional
        Spread of sigma points around the mean (default 1e-3).
    beta : float, optional
        Prior distribution parameter; 2 is optimal for Gaussian (default 2.0).
    kappa : float, optional
        Secondary scaling parameter (default 0).

    Examples
    --------
    >>> import numpy as np
    >>> from tfilterspy import UnscentedKalmanFilter
    >>> def f(x): return x
    >>> def h(x): return x[:1]
    >>> ukf = UnscentedKalmanFilter(f, h,
    ...     Q=np.eye(2)*0.1, R=np.eye(1)*0.5,
    ...     x0=np.zeros(2), P0=np.eye(2))
    >>> ukf.fit(np.random.randn(50, 1))
    >>> states = ukf.predict()
    """

    def __init__(self, f, h, Q, R, x0, P0, alpha=1e-3, beta=2.0, kappa=0.0):
        super().__init__()
        self.f = f
        self.h = h
        self.Q = np.asarray(Q, dtype=np.float64)
        self.R = np.asarray(R, dtype=np.float64)
        self.x0 = np.asarray(x0, dtype=np.float64)
        self.P0 = np.asarray(P0, dtype=np.float64)
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa
        self.n_state = len(x0)
        self.n_obs = self.R.shape[0]
        self.is_fitted_ = False
        self._compute_weights()

    def _compute_weights(self):
        n = self.n_state
        lam = self.alpha ** 2 * (n + self.kappa) - n
        self._lambda = lam

        self._Wm = np.full(2 * n + 1, 1.0 / (2.0 * (n + lam)))
        self._Wc = np.full(2 * n + 1, 1.0 / (2.0 * (n + lam)))
        self._Wm[0] = lam / (n + lam)
        self._Wc[0] = lam / (n + lam) + (1 - self.alpha ** 2 + self.beta)

    def _sigma_points(self, x, P):
        n = self.n_state
        scale = n + self._lambda
        try:
            sqrt_P = linalg.cholesky(scale * P, lower=True)
        except linalg.LinAlgError:
            sqrt_P = linalg.cholesky(scale * P + 1e-6 * np.eye(n), lower=True)

        sigmas = np.empty((2 * n + 1, n))
        sigmas[0] = x
        for i in range(n):
            sigmas[i + 1] = x + sqrt_P[:, i]
            sigmas[n + i + 1] = x - sqrt_P[:, i]
        return sigmas

    def _unscented_transform(self, sigmas, func, noise_cov):
        transformed = np.array([func(s) for s in sigmas])
        dim = transformed.shape[1]

        mean = self._Wm @ transformed
        diff = transformed - mean
        cov = np.zeros((dim, dim))
        for i in range(len(sigmas)):
            cov += self._Wc[i] * np.outer(diff[i], diff[i])
        cov += noise_cov

        return mean, cov, transformed

    def fit(self, X):
        """
        Run the UKF forward pass.

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

        self.filtered_states_ = np.empty((n, ns))
        self.filtered_covs_ = np.empty((n, ns, ns))
        self.predicted_states_ = np.empty((n, ns))
        self.predicted_covs_ = np.empty((n, ns, ns))
        self.log_likelihood_ = 0.0

        x = self.x0.copy()
        P = self.P0.copy()

        for i in range(n):
            # --- Predict via unscented transform ---
            sigmas = self._sigma_points(x, P)
            x_pred, P_pred, sigmas_f = self._unscented_transform(
                sigmas, self.f, self.Q
            )

            self.predicted_states_[i] = x_pred
            self.predicted_covs_[i] = P_pred

            # --- Observation sigma points ---
            sigmas_pred = self._sigma_points(x_pred, P_pred)
            z_pred, S, sigmas_h = self._unscented_transform(
                sigmas_pred, self.h, self.R
            )

            # --- Cross-covariance ---
            Pxz = np.zeros((ns, no))
            for j in range(2 * ns + 1):
                Pxz += self._Wc[j] * np.outer(
                    sigmas_pred[j] - x_pred, sigmas_h[j] - z_pred
                )

            # --- Kalman gain and update ---
            S_inv = linalg.solve(S, np.eye(no), assume_a="pos")
            K = Pxz @ S_inv

            y = X[i] - z_pred
            x = x_pred + K @ y
            P = P_pred - K @ S @ K.T
            P = 0.5 * (P + P.T)

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

    def filter_step(self, z):
        """Single-step online UKF update."""
        if not hasattr(self, "_online_x"):
            self._online_x = self.x0.copy()
            self._online_P = self.P0.copy()

        z = np.asarray(z, dtype=np.float64)
        ns = self.n_state
        no = self.n_obs

        sigmas = self._sigma_points(self._online_x, self._online_P)
        x_pred, P_pred, _ = self._unscented_transform(sigmas, self.f, self.Q)

        sigmas_pred = self._sigma_points(x_pred, P_pred)
        z_pred, S, sigmas_h = self._unscented_transform(sigmas_pred, self.h, self.R)

        Pxz = np.zeros((ns, no))
        for j in range(2 * ns + 1):
            Pxz += self._Wc[j] * np.outer(
                sigmas_pred[j] - x_pred, sigmas_h[j] - z_pred
            )

        K = Pxz @ linalg.inv(S)
        y = z - z_pred
        self._online_x = x_pred + K @ y
        self._online_P = P_pred - K @ S @ K.T
        self._online_P = 0.5 * (self._online_P + self._online_P.T)

        return self._online_x.copy(), self._online_P.copy()
