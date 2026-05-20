import numpy as np
import dask
from scipy import linalg

from tfilterspy.base_estimator import BaseEstimator


class ParticleFilter(BaseEstimator):
    r"""
    Sequential Importance Resampling (SIR) Particle Filter.

    Represents the posterior distribution using a set of weighted particles.
    Handles nonlinear, non-Gaussian state-space models where Kalman-based
    approaches fail.

    Supports both callable and matrix-based transition/observation models.
    When matrices are provided, particle propagation is vectorized for speed.

    State-space model::

        x_{k+1} = f(x_k) + w_k,    w_k ~ N(0, Q)
        z_k     = h(x_k) + v_k,     v_k ~ N(0, R)

    Parameters
    ----------
    f : callable or ndarray
        State transition. If callable: ``f(x) -> x_next``.
        If ndarray (n_state, n_state): linear transition matrix.
    h : callable or ndarray
        Observation model. If callable: ``h(x) -> z``.
        If ndarray (n_obs, n_state): linear observation matrix.
    Q : ndarray, shape (n_state, n_state)
        Process noise covariance.
    R : ndarray, shape (n_obs, n_obs)
        Observation noise covariance.
    x0 : ndarray, shape (n_state,)
        Initial state estimate.
    n_particles : int, optional
        Number of particles (default 1000).
    resample_threshold : float, optional
        ESS ratio below which resampling triggers (default 0.5).
    use_dask : bool, optional
        Parallelize particle propagation with Dask (default False).

    Examples
    --------
    >>> import numpy as np
    >>> from tfilterspy import ParticleFilter
    >>> F = np.eye(2)
    >>> H = np.eye(2)
    >>> pf = ParticleFilter(F, H, Q=np.eye(2)*0.1, R=np.eye(2)*0.5,
    ...     x0=np.zeros(2), n_particles=500)
    >>> pf.fit(np.random.randn(100, 2))
    >>> states = pf.predict()
    """

    def __init__(self, f, h, Q, R, x0, n_particles=1000,
                 resample_threshold=0.5, use_dask=False):
        super().__init__()
        self.Q = np.asarray(Q, dtype=np.float64)
        self.R = np.asarray(R, dtype=np.float64)
        self.x0 = np.asarray(x0, dtype=np.float64)
        self.n_particles = n_particles
        self.resample_threshold = resample_threshold
        self.use_dask = use_dask
        self.n_state = len(self.x0)
        self.n_obs = self.R.shape[0]
        self.is_fitted_ = False

        # Support both callable and matrix-based models
        if isinstance(f, np.ndarray):
            self._F_matrix = np.asarray(f, dtype=np.float64)
            self.f = lambda x: self._F_matrix @ x
            self._f_vectorized = True
        else:
            self._F_matrix = None
            self.f = f
            self._f_vectorized = False

        if isinstance(h, np.ndarray):
            self._H_matrix = np.asarray(h, dtype=np.float64)
            self.h = lambda x: self._H_matrix @ x
            self._h_vectorized = True
        else:
            self._H_matrix = None
            self.h = h
            self._h_vectorized = False

    def _init_particles(self):
        self._particles = np.tile(self.x0, (self.n_particles, 1))
        self._weights = np.ones(self.n_particles) / self.n_particles

    def _propagate(self):
        """Propagate particles through state transition + process noise."""
        L_Q = linalg.cholesky(self.Q + 1e-10 * np.eye(self.n_state), lower=True)
        noise = (L_Q @ np.random.randn(self.n_state, self.n_particles)).T

        if self._f_vectorized:
            self._particles = self._particles @ self._F_matrix.T + noise
        elif self.use_dask:
            delayed = [dask.delayed(self.f)(self._particles[j])
                       for j in range(self.n_particles)]
            self._particles = np.array(dask.compute(*delayed)) + noise
        else:
            self._particles = np.array(
                [self.f(self._particles[j]) for j in range(self.n_particles)]
            ) + noise

    def _compute_predicted_obs(self):
        """Compute predicted observations for all particles."""
        if self._h_vectorized:
            return self._particles @ self._H_matrix.T
        elif self.use_dask:
            delayed = [dask.delayed(self.h)(self._particles[j])
                       for j in range(self.n_particles)]
            return np.array(dask.compute(*delayed))
        else:
            return np.array(
                [self.h(self._particles[j]) for j in range(self.n_particles)]
            )

    def _update_weights(self, z, predicted_obs):
        """Update particle weights based on measurement likelihood."""
        R_inv = linalg.inv(self.R)
        _, logdet_R = np.linalg.slogdet(self.R)
        log_norm = -0.5 * (self.n_obs * np.log(2 * np.pi) + logdet_R)

        diff = predicted_obs - z
        # Vectorized log-likelihood for all particles
        log_likelihoods = log_norm - 0.5 * np.sum(diff @ R_inv * diff, axis=1)

        # Numerical stability: subtract max before exp
        log_likelihoods -= log_likelihoods.max()
        likelihoods = np.exp(log_likelihoods)

        self._weights *= likelihoods
        self._weights += 1e-300
        self._weights /= self._weights.sum()

    def _systematic_resample(self):
        """Systematic resampling — lower variance than multinomial."""
        N = self.n_particles
        positions = (np.arange(N) + np.random.uniform()) / N
        cumsum = np.cumsum(self._weights)
        cumsum[-1] = 1.0  # ensure no floating-point overshoot
        indices = np.searchsorted(cumsum, positions)
        self._particles = self._particles[indices].copy()
        self._weights = np.ones(N) / N

    def _effective_sample_size(self):
        return 1.0 / np.sum(self._weights ** 2)

    def fit(self, X):
        """
        Run the particle filter on measurements.

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
        self._init_particles()

        n = len(X)
        ns = self.n_state

        self.filtered_states_ = np.empty((n, ns))
        self.filtered_covs_ = np.empty((n, ns, ns))
        self.effective_sample_sizes_ = np.empty(n)

        for i in range(n):
            self._propagate()

            predicted_obs = self._compute_predicted_obs()
            self._update_weights(X[i], predicted_obs)

            # State estimate
            self.filtered_states_[i] = self._weights @ self._particles
            diff = self._particles - self.filtered_states_[i]
            self.filtered_covs_[i] = (diff * self._weights[:, None]).T @ diff

            # ESS-based resampling
            ess = self._effective_sample_size()
            self.effective_sample_sizes_[i] = ess
            if ess < self.resample_threshold * self.n_particles:
                self._systematic_resample()

        self.is_fitted_ = True
        return self

    def predict(self):
        """Return filtered state estimates."""
        self._check_fitted()
        return self.filtered_states_

    def filter_step(self, z):
        """
        Single-step online particle filter update.

        Parameters
        ----------
        z : ndarray, shape (n_obs,)
            Single measurement.

        Returns
        -------
        state : ndarray, shape (n_state,)
            Weighted mean state estimate.
        """
        if not hasattr(self, "_online_init"):
            self._init_particles()
            self._online_init = True

        z = np.asarray(z, dtype=np.float64)

        self._propagate()
        predicted_obs = self._compute_predicted_obs()
        self._update_weights(z, predicted_obs)

        state = self._weights @ self._particles

        ess = self._effective_sample_size()
        if ess < self.resample_threshold * self.n_particles:
            self._systematic_resample()

        return state


class DaskParticleFilter(ParticleFilter):
    """
    Backward-compatible Particle Filter wrapper.

    Accepts the original DaskParticleFilter constructor signature
    and delegates to the new ParticleFilter implementation.
    """

    def __init__(self, state_transition, observation_model, process_noise_cov,
                 observation_noise_cov, initial_state, num_particles=1000,
                 use_dask=True, estimation_strategy="residual_analysis"):
        super().__init__(
            f=np.asarray(state_transition, dtype=np.float64),
            h=np.asarray(observation_model, dtype=np.float64),
            Q=process_noise_cov,
            R=observation_noise_cov,
            x0=initial_state,
            n_particles=num_particles,
            use_dask=use_dask,
        )
        self.estimation_strategy = estimation_strategy
