import numpy as np
import dask
from scipy import linalg

from tfilterspy.base_estimator import BaseEstimator


class EnsembleKalmanFilter(BaseEstimator):
    r"""
    Ensemble Kalman Filter for high-dimensional state-space models.

    Approximates the Kalman filter using a Monte Carlo ensemble of state
    realizations. Scales to systems with millions of state variables where
    storing full covariance matrices is infeasible.

    Uses Dask for parallel ensemble member propagation when ``use_dask=True``.

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
    n_ensemble : int, optional
        Number of ensemble members (default 100).
    use_dask : bool, optional
        Use Dask for parallel ensemble computation (default True).

    Examples
    --------
    >>> import numpy as np
    >>> from tfilterspy import EnsembleKalmanFilter
    >>> def f(x): return x
    >>> def h(x): return x[:1]
    >>> enkf = EnsembleKalmanFilter(f, h,
    ...     Q=np.eye(2)*0.1, R=np.eye(1)*0.5,
    ...     x0=np.zeros(2), P0=np.eye(2), n_ensemble=50)
    >>> enkf.fit(np.random.randn(50, 1))
    >>> states = enkf.predict()
    """

    def __init__(self, f, h, Q, R, x0, P0, n_ensemble=100, use_dask=True):
        super().__init__()
        self.f = f
        self.h = h
        self.Q = np.asarray(Q, dtype=np.float64)
        self.R = np.asarray(R, dtype=np.float64)
        self.x0 = np.asarray(x0, dtype=np.float64)
        self.P0 = np.asarray(P0, dtype=np.float64)
        self.n_ensemble = n_ensemble
        self.use_dask = use_dask
        self.n_state = len(x0)
        self.n_obs = self.R.shape[0]
        self.is_fitted_ = False

    def _init_ensemble(self):
        L = linalg.cholesky(self.P0 + 1e-10 * np.eye(self.n_state), lower=True)
        noise = np.random.randn(self.n_state, self.n_ensemble)
        self.ensemble_ = self.x0[:, None] + L @ noise  # (n_state, n_ensemble)

    def _propagate_ensemble(self, ensemble):
        ne = ensemble.shape[1]
        if self.use_dask:
            delayed_preds = [
                dask.delayed(self.f)(ensemble[:, j]) for j in range(ne)
            ]
            results = dask.compute(*delayed_preds)
            return np.column_stack(results)
        else:
            return np.column_stack([self.f(ensemble[:, j]) for j in range(ne)])

    def _observe_ensemble(self, ensemble):
        ne = ensemble.shape[1]
        if self.use_dask:
            delayed_obs = [
                dask.delayed(self.h)(ensemble[:, j]) for j in range(ne)
            ]
            results = dask.compute(*delayed_obs)
            return np.column_stack(results)
        else:
            return np.column_stack([self.h(ensemble[:, j]) for j in range(ne)])

    def fit(self, X):
        """
        Run the EnKF forward pass.

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
        self._init_ensemble()

        n = len(X)
        ns = self.n_state
        ne = self.n_ensemble
        no = self.n_obs

        self.filtered_states_ = np.empty((n, ns))
        self.filtered_covs_ = np.empty((n, ns, ns))

        L_Q = linalg.cholesky(self.Q + 1e-10 * np.eye(ns), lower=True)
        L_R = linalg.cholesky(self.R + 1e-10 * np.eye(no), lower=True)

        ensemble = self.ensemble_.copy()

        for i in range(n):
            # --- Predict: propagate + add process noise ---
            predicted = self._propagate_ensemble(ensemble)
            process_noise = L_Q @ np.random.randn(ns, ne)
            predicted += process_noise

            # --- Ensemble statistics ---
            x_mean = predicted.mean(axis=1)
            A = (predicted - x_mean[:, None]) / np.sqrt(ne - 1)

            # --- Predicted observations ---
            pred_obs = self._observe_ensemble(predicted)
            z_mean = pred_obs.mean(axis=1)
            HA = (pred_obs - z_mean[:, None]) / np.sqrt(ne - 1)

            # --- Perturbed observations ---
            obs_noise = L_R @ np.random.randn(no, ne)
            perturbed_obs = X[i][:, None] + obs_noise

            # --- EnKF update ---
            S = HA @ HA.T + self.R
            K = A @ HA.T @ linalg.inv(S)

            innovations = perturbed_obs - pred_obs
            ensemble = predicted + K @ innovations

            # --- Store results ---
            self.filtered_states_[i] = ensemble.mean(axis=1)
            diff = ensemble - self.filtered_states_[i][:, None]
            self.filtered_covs_[i] = (diff @ diff.T) / (ne - 1)

        self.ensemble_ = ensemble
        self.is_fitted_ = True
        return self

    def predict(self):
        """Return filtered state estimates."""
        self._check_fitted()
        return self.filtered_states_

    def filter_step(self, z):
        """Single-step online EnKF update."""
        if not hasattr(self, "_online_ensemble"):
            self._init_ensemble()
            self._online_ensemble = self.ensemble_.copy()

        z = np.asarray(z, dtype=np.float64)
        ns = self.n_state
        ne = self.n_ensemble
        no = self.n_obs

        L_Q = linalg.cholesky(self.Q + 1e-10 * np.eye(ns), lower=True)
        L_R = linalg.cholesky(self.R + 1e-10 * np.eye(no), lower=True)

        predicted = self._propagate_ensemble(self._online_ensemble)
        predicted += L_Q @ np.random.randn(ns, ne)

        x_mean = predicted.mean(axis=1)
        A = (predicted - x_mean[:, None]) / np.sqrt(ne - 1)

        pred_obs = self._observe_ensemble(predicted)
        z_mean = pred_obs.mean(axis=1)
        HA = (pred_obs - z_mean[:, None]) / np.sqrt(ne - 1)

        S = HA @ HA.T + self.R
        K = A @ HA.T @ linalg.inv(S)

        perturbed_obs = z[:, None] + L_R @ np.random.randn(no, ne)
        innovations = perturbed_obs - pred_obs
        self._online_ensemble = predicted + K @ innovations

        state = self._online_ensemble.mean(axis=1)
        return state
