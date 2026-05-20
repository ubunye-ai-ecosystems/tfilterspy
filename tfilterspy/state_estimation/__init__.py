"""
State Estimation submodule containing implementations for:
- KalmanFilter (linear Gaussian systems)
- ExtendedKalmanFilter (nonlinear, Jacobian-based)
- UnscentedKalmanFilter (nonlinear, sigma points)
- EnsembleKalmanFilter (high-dimensional, Dask-parallel)
- ParticleFilter (nonlinear, non-Gaussian)
"""
from .linear_filters import KalmanFilter, DaskKalmanFilter
from .extended_kalman import ExtendedKalmanFilter
from .unscented_kalman import UnscentedKalmanFilter
from .ensemble_kalman import EnsembleKalmanFilter
from .particle_filters import ParticleFilter, DaskParticleFilter

# Backward compat
from .nonlinear_filters import DaskNonLinearKalmanFilter

__all__ = [
    "KalmanFilter",
    "ExtendedKalmanFilter",
    "UnscentedKalmanFilter",
    "EnsembleKalmanFilter",
    "ParticleFilter",
    "DaskKalmanFilter",
    "DaskParticleFilter",
    "DaskNonLinearKalmanFilter",
]
