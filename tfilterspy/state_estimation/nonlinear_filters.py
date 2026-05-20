"""
Nonlinear filter re-exports.

Provides ExtendedKalmanFilter and UnscentedKalmanFilter under the
nonlinear_filters namespace for backward compatibility.
"""
from .extended_kalman import ExtendedKalmanFilter
from .unscented_kalman import UnscentedKalmanFilter

DaskNonLinearKalmanFilter = ExtendedKalmanFilter

__all__ = [
    "ExtendedKalmanFilter",
    "UnscentedKalmanFilter",
    "DaskNonLinearKalmanFilter",
]
