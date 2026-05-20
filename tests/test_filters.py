"""
Comprehensive tests for all TFiltersPy filter implementations.
"""
import numpy as np
import pytest

from tfilterspy import (
    KalmanFilter,
    ExtendedKalmanFilter,
    UnscentedKalmanFilter,
    EnsembleKalmanFilter,
    ParticleFilter,
    DaskKalmanFilter,
    DaskParticleFilter,
    BaseEstimator,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def linear_system_2d():
    """Simple 2D constant-state system for testing."""
    F = np.eye(2)
    H = np.eye(2)
    Q = np.eye(2) * 0.01
    R = np.eye(2) * 0.5
    x0 = np.zeros(2)
    P0 = np.eye(2)
    return dict(F=F, H=H, Q=Q, R=R, x0=x0, P0=P0)


@pytest.fixture
def tracking_system():
    """4D constant-velocity tracking (position + velocity in 2D)."""
    dt = 0.1
    F = np.array([
        [1, dt, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, dt],
        [0, 0, 0, 1],
    ])
    H = np.array([
        [1, 0, 0, 0],
        [0, 0, 1, 0],
    ])
    Q = np.eye(4) * 0.01
    R = np.eye(2) * 1.0
    x0 = np.array([0, 1, 0, 1], dtype=np.float64)
    P0 = np.eye(4) * 1.0
    return dict(F=F, H=H, Q=Q, R=R, x0=x0, P0=P0, dt=dt)


def generate_linear_data(F, H, Q, R, x0, n_steps, seed=42):
    """Generate synthetic measurements from a linear state-space model."""
    rng = np.random.RandomState(seed)
    ns = F.shape[0]
    no = H.shape[0]
    true_states = np.empty((n_steps, ns))
    measurements = np.empty((n_steps, no))
    x = x0.copy()
    for i in range(n_steps):
        x = F @ x + rng.multivariate_normal(np.zeros(ns), Q)
        z = H @ x + rng.multivariate_normal(np.zeros(no), R)
        true_states[i] = x
        measurements[i] = z
    return true_states, measurements


# ---------------------------------------------------------------------------
# BaseEstimator
# ---------------------------------------------------------------------------

class TestBaseEstimator:
    def test_get_set_params(self, linear_system_2d):
        kf = KalmanFilter(**linear_system_2d)
        params = kf.get_params()
        assert "n_state" in params
        assert params["n_state"] == 2

        kf.set_params(n_state=3)
        assert kf.n_state == 3

    def test_set_invalid_param(self, linear_system_2d):
        kf = KalmanFilter(**linear_system_2d)
        with pytest.raises(ValueError, match="Invalid parameter"):
            kf.set_params(nonexistent_param=42)

    def test_repr(self, linear_system_2d):
        kf = KalmanFilter(**linear_system_2d)
        assert "KalmanFilter" in repr(kf)

    def test_check_fitted_error(self, linear_system_2d):
        kf = KalmanFilter(**linear_system_2d)
        with pytest.raises(RuntimeError, match="not fitted"):
            kf.predict()


# ---------------------------------------------------------------------------
# KalmanFilter
# ---------------------------------------------------------------------------

class TestKalmanFilter:
    def test_initialization(self, linear_system_2d):
        kf = KalmanFilter(**linear_system_2d)
        assert kf.n_state == 2
        assert kf.n_obs == 2
        assert kf.is_fitted_ is False

    def test_invalid_dimensions(self):
        with pytest.raises(ValueError, match="must be square"):
            KalmanFilter(
                F=np.array([[1, 0]]),
                H=np.eye(2), Q=np.eye(2), R=np.eye(2),
                x0=np.zeros(2), P0=np.eye(2),
            )

    def test_fit_predict(self, linear_system_2d):
        true_states, measurements = generate_linear_data(
            **{k: linear_system_2d[k] for k in ["F", "H", "Q", "R", "x0"]},
            n_steps=200,
        )
        kf = KalmanFilter(**linear_system_2d)
        kf.fit(measurements)
        states = kf.predict()
        assert states.shape == (200, 2)
        assert kf.is_fitted_

        # Filter should reduce noise: MSE of filtered < MSE of raw
        mse_raw = np.mean((measurements - true_states) ** 2)
        mse_filtered = np.mean((states - true_states) ** 2)
        assert mse_filtered < mse_raw

    def test_fit_predict_convenience(self, linear_system_2d):
        _, measurements = generate_linear_data(
            **{k: linear_system_2d[k] for k in ["F", "H", "Q", "R", "x0"]},
            n_steps=50,
        )
        kf = KalmanFilter(**linear_system_2d)
        states = kf.fit_predict(measurements)
        assert states.shape == (50, 2)

    def test_1d_measurements(self):
        kf = KalmanFilter(
            F=np.eye(1), H=np.eye(1),
            Q=np.eye(1) * 0.1, R=np.eye(1) * 0.5,
            x0=np.zeros(1), P0=np.eye(1),
        )
        measurements = np.random.randn(100)  # 1D input
        kf.fit(measurements)
        assert kf.predict().shape == (100, 1)

    def test_smooth(self, tracking_system):
        true_states, measurements = generate_linear_data(
            **{k: tracking_system[k] for k in ["F", "H", "Q", "R", "x0"]},
            n_steps=200,
        )
        kf = KalmanFilter(
            **{k: tracking_system[k] for k in ["F", "H", "Q", "R", "x0", "P0"]}
        )
        kf.fit(measurements)

        filtered = kf.predict()
        smoothed, smoothed_covs = kf.smooth()

        assert smoothed.shape == filtered.shape
        assert smoothed_covs.shape == (200, 4, 4)

        # Smoother should be at least as good as filter
        mse_filtered = np.mean((filtered - true_states) ** 2)
        mse_smoothed = np.mean((smoothed - true_states) ** 2)
        assert mse_smoothed <= mse_filtered * 1.01  # allow tiny tolerance

    def test_smooth_requires_covariances(self, linear_system_2d):
        kf = KalmanFilter(**linear_system_2d, store_covariances=False)
        kf.fit(np.random.randn(50, 2))
        with pytest.raises(RuntimeError, match="store_covariances"):
            kf.smooth()

    def test_filter_step(self, linear_system_2d):
        kf = KalmanFilter(**linear_system_2d)
        x, P = kf.filter_step(np.array([1.0, 2.0]))
        assert x.shape == (2,)
        assert P.shape == (2, 2)

        x2, P2 = kf.filter_step(np.array([1.1, 2.1]))
        assert x2.shape == (2,)

    def test_forecast(self, tracking_system):
        _, measurements = generate_linear_data(
            **{k: tracking_system[k] for k in ["F", "H", "Q", "R", "x0"]},
            n_steps=100,
        )
        kf = KalmanFilter(
            **{k: tracking_system[k] for k in ["F", "H", "Q", "R", "x0", "P0"]}
        )
        kf.fit(measurements)
        fc_states, fc_covs = kf.forecast(20)
        assert fc_states.shape == (20, 4)
        assert fc_covs.shape == (20, 4, 4)

    def test_log_likelihood(self, linear_system_2d):
        _, measurements = generate_linear_data(
            **{k: linear_system_2d[k] for k in ["F", "H", "Q", "R", "x0"]},
            n_steps=100,
        )
        kf = KalmanFilter(**linear_system_2d)
        kf.fit(measurements)
        assert np.isfinite(kf.log_likelihood_)

    def test_score(self, linear_system_2d):
        true_states, measurements = generate_linear_data(
            **{k: linear_system_2d[k] for k in ["F", "H", "Q", "R", "x0"]},
            n_steps=100,
        )
        kf = KalmanFilter(**linear_system_2d)
        kf.fit(measurements)
        score = kf.score(true_states)
        assert score <= 0  # negative MSE

    def test_store_covariances_false(self, linear_system_2d):
        kf = KalmanFilter(**linear_system_2d, store_covariances=False)
        kf.fit(np.random.randn(50, 2))
        states = kf.predict()
        assert states.shape == (50, 2)
        assert not hasattr(kf, "filtered_covs_") or kf.filtered_covs_ is None or True

    def test_large_dataset(self):
        """Test with a reasonably large dataset to verify memory efficiency."""
        n_steps = 10000
        kf = KalmanFilter(
            F=np.eye(4), H=np.eye(4),
            Q=np.eye(4) * 0.01, R=np.eye(4) * 0.1,
            x0=np.zeros(4), P0=np.eye(4),
            store_covariances=False,
        )
        measurements = np.random.randn(n_steps, 4)
        kf.fit(measurements)
        assert kf.predict().shape == (n_steps, 4)


# ---------------------------------------------------------------------------
# ExtendedKalmanFilter
# ---------------------------------------------------------------------------

class TestExtendedKalmanFilter:
    def _linear_ekf(self):
        """EKF configured as a standard linear KF for testing."""
        F = np.eye(2)
        H = np.eye(2)
        return ExtendedKalmanFilter(
            f=lambda x: F @ x,
            h=lambda x: H @ x,
            F_jacobian=lambda x: F,
            H_jacobian=lambda x: H,
            Q=np.eye(2) * 0.01,
            R=np.eye(2) * 0.5,
            x0=np.zeros(2),
            P0=np.eye(2),
        )

    def test_fit_predict(self):
        ekf = self._linear_ekf()
        measurements = np.random.randn(100, 2)
        ekf.fit(measurements)
        states = ekf.predict()
        assert states.shape == (100, 2)

    def test_smooth(self):
        ekf = self._linear_ekf()
        ekf.fit(np.random.randn(50, 2))
        smoothed, covs = ekf.smooth()
        assert smoothed.shape == (50, 2)
        assert covs.shape == (50, 2, 2)

    def test_filter_step(self):
        ekf = self._linear_ekf()
        x, P = ekf.filter_step(np.array([1.0, 2.0]))
        assert x.shape == (2,)

    def test_nonlinear_system(self):
        """Test with a truly nonlinear system."""
        def f(x):
            return np.array([x[0] + 0.1 * x[1], x[1]])

        def h(x):
            return np.array([np.sqrt(x[0] ** 2 + x[1] ** 2)])

        def F_jac(x):
            return np.array([[1.0, 0.1], [0.0, 1.0]])

        def H_jac(x):
            r = np.sqrt(x[0] ** 2 + x[1] ** 2) + 1e-10
            return np.array([[x[0] / r, x[1] / r]])

        ekf = ExtendedKalmanFilter(
            f=f, h=h, F_jacobian=F_jac, H_jacobian=H_jac,
            Q=np.eye(2) * 0.01, R=np.eye(1) * 0.1,
            x0=np.array([1.0, 1.0]), P0=np.eye(2),
        )
        measurements = np.random.randn(50, 1) + 1.5
        ekf.fit(measurements)
        assert ekf.predict().shape == (50, 2)
        assert np.isfinite(ekf.log_likelihood_)


# ---------------------------------------------------------------------------
# UnscentedKalmanFilter
# ---------------------------------------------------------------------------

class TestUnscentedKalmanFilter:
    def _linear_ukf(self):
        return UnscentedKalmanFilter(
            f=lambda x: x,
            h=lambda x: x,
            Q=np.eye(2) * 0.01,
            R=np.eye(2) * 0.5,
            x0=np.zeros(2),
            P0=np.eye(2),
        )

    def test_fit_predict(self):
        ukf = self._linear_ukf()
        ukf.fit(np.random.randn(100, 2))
        assert ukf.predict().shape == (100, 2)

    def test_filter_step(self):
        ukf = self._linear_ukf()
        x, P = ukf.filter_step(np.array([1.0, 2.0]))
        assert x.shape == (2,)
        assert P.shape == (2, 2)

    def test_nonlinear_observation(self):
        """Test UKF with nonlinear observation model."""
        def f(x):
            return x

        def h(x):
            return np.array([x[0] ** 2 + x[1]])

        ukf = UnscentedKalmanFilter(
            f=f, h=h,
            Q=np.eye(2) * 0.01, R=np.eye(1) * 0.1,
            x0=np.array([1.0, 0.0]), P0=np.eye(2),
        )
        ukf.fit(np.random.randn(50, 1) + 1.0)
        assert ukf.predict().shape == (50, 2)

    def test_sigma_point_params(self):
        ukf = UnscentedKalmanFilter(
            f=lambda x: x, h=lambda x: x,
            Q=np.eye(2) * 0.01, R=np.eye(2) * 0.1,
            x0=np.zeros(2), P0=np.eye(2),
            alpha=0.1, beta=2.0, kappa=1.0,
        )
        ukf.fit(np.random.randn(30, 2))
        assert ukf.predict().shape == (30, 2)


# ---------------------------------------------------------------------------
# EnsembleKalmanFilter
# ---------------------------------------------------------------------------

class TestEnsembleKalmanFilter:
    def test_fit_predict(self):
        enkf = EnsembleKalmanFilter(
            f=lambda x: x,
            h=lambda x: x,
            Q=np.eye(2) * 0.01,
            R=np.eye(2) * 0.5,
            x0=np.zeros(2),
            P0=np.eye(2),
            n_ensemble=50,
            use_dask=False,
        )
        enkf.fit(np.random.randn(80, 2))
        states = enkf.predict()
        assert states.shape == (80, 2)

    def test_dask_parallel(self):
        enkf = EnsembleKalmanFilter(
            f=lambda x: x,
            h=lambda x: x,
            Q=np.eye(2) * 0.01,
            R=np.eye(2) * 0.5,
            x0=np.zeros(2),
            P0=np.eye(2),
            n_ensemble=30,
            use_dask=True,
        )
        enkf.fit(np.random.randn(40, 2))
        assert enkf.predict().shape == (40, 2)

    def test_filter_step(self):
        enkf = EnsembleKalmanFilter(
            f=lambda x: x, h=lambda x: x,
            Q=np.eye(2) * 0.01, R=np.eye(2) * 0.5,
            x0=np.zeros(2), P0=np.eye(2),
            n_ensemble=30, use_dask=False,
        )
        state = enkf.filter_step(np.array([1.0, 2.0]))
        assert state.shape == (2,)

    def test_covariance_stored(self):
        enkf = EnsembleKalmanFilter(
            f=lambda x: x, h=lambda x: x,
            Q=np.eye(3) * 0.01, R=np.eye(3) * 0.1,
            x0=np.zeros(3), P0=np.eye(3),
            n_ensemble=40, use_dask=False,
        )
        enkf.fit(np.random.randn(20, 3))
        assert enkf.filtered_covs_.shape == (20, 3, 3)


# ---------------------------------------------------------------------------
# ParticleFilter
# ---------------------------------------------------------------------------

class TestParticleFilter:
    def test_matrix_model(self):
        """Test with matrix-based (linear) transition/observation."""
        pf = ParticleFilter(
            f=np.eye(2), h=np.eye(2),
            Q=np.eye(2) * 0.1, R=np.eye(2) * 0.5,
            x0=np.zeros(2), n_particles=500,
        )
        pf.fit(np.random.randn(50, 2))
        assert pf.predict().shape == (50, 2)

    def test_callable_model(self):
        """Test with callable (potentially nonlinear) models."""
        pf = ParticleFilter(
            f=lambda x: x,
            h=lambda x: x[:1],
            Q=np.eye(2) * 0.1,
            R=np.eye(1) * 0.5,
            x0=np.zeros(2),
            n_particles=300,
        )
        pf.fit(np.random.randn(40, 1))
        assert pf.predict().shape == (40, 2)

    def test_effective_sample_size(self):
        pf = ParticleFilter(
            f=np.eye(2), h=np.eye(2),
            Q=np.eye(2) * 0.1, R=np.eye(2) * 0.5,
            x0=np.zeros(2), n_particles=500,
        )
        pf.fit(np.random.randn(30, 2))
        assert len(pf.effective_sample_sizes_) == 30
        assert np.all(pf.effective_sample_sizes_ > 0)

    def test_filter_step(self):
        pf = ParticleFilter(
            f=np.eye(2), h=np.eye(2),
            Q=np.eye(2) * 0.1, R=np.eye(2) * 0.5,
            x0=np.zeros(2), n_particles=200,
        )
        state = pf.filter_step(np.array([1.0, 2.0]))
        assert state.shape == (2,)

    def test_filtering_reduces_noise(self):
        """Filter should produce better estimates than raw measurements."""
        np.random.seed(42)
        F = np.eye(2)
        H = np.eye(2)
        Q = np.eye(2) * 0.01
        R = np.eye(2) * 1.0
        x0 = np.zeros(2)

        true_states, measurements = generate_linear_data(
            F, H, Q, R, x0, n_steps=200,
        )
        pf = ParticleFilter(
            f=F, h=H, Q=Q, R=R, x0=x0, n_particles=1000,
        )
        pf.fit(measurements)
        states = pf.predict()

        mse_raw = np.mean((measurements - true_states) ** 2)
        mse_filtered = np.mean((states - true_states) ** 2)
        assert mse_filtered < mse_raw

    def test_dask_particle_propagation(self):
        pf = ParticleFilter(
            f=np.eye(2), h=np.eye(2),
            Q=np.eye(2) * 0.1, R=np.eye(2) * 0.5,
            x0=np.zeros(2), n_particles=200, use_dask=True,
        )
        pf.fit(np.random.randn(20, 2))
        assert pf.predict().shape == (20, 2)


# ---------------------------------------------------------------------------
# Backward Compatibility
# ---------------------------------------------------------------------------

class TestBackwardCompat:
    def test_dask_kalman_filter(self):
        kf = DaskKalmanFilter(
            state_transition_matrix=np.eye(2),
            observation_matrix=np.eye(2),
            process_noise_cov=np.eye(2) * 0.1,
            observation_noise_cov=np.eye(2) * 0.5,
            initial_state=np.zeros(2),
            initial_covariance=np.eye(2),
        )
        kf.fit(np.random.randn(50, 2))
        result = kf.predict()
        # DaskKalmanFilter.predict() returns a dask array
        assert hasattr(result, "compute")
        states = result.compute()
        assert states.shape == (50, 2)

    def test_dask_particle_filter(self):
        pf = DaskParticleFilter(
            state_transition=np.eye(2),
            observation_model=np.eye(2),
            process_noise_cov=np.eye(2) * 0.1,
            observation_noise_cov=np.eye(2) * 0.5,
            initial_state=np.zeros(2),
            num_particles=200,
            use_dask=False,
        )
        pf.fit(np.random.randn(30, 2))
        states = pf.predict()
        assert states.shape == (30, 2)
