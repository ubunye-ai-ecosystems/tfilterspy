import numpy as np
import dask.array as da


class BaseEstimator:
    """
    Base class for all estimators in TFiltersPy.

    Provides sklearn-compatible parameter handling, Dask array conversion,
    and shared methods for all filter implementations.
    """

    def __init__(self, name=None):
        self.name = name or self.__class__.__name__

    @staticmethod
    def to_dask_array(numpy_array, chunk_size=None):
        """Convert a NumPy array to a Dask array with specified chunking."""
        if chunk_size is None:
            return da.from_array(numpy_array, chunks="auto")
        if isinstance(chunk_size, int):
            chunks = tuple(chunk_size for _ in range(numpy_array.ndim))
        else:
            chunks = chunk_size
        return da.from_array(numpy_array, chunks=chunks)

    def get_params(self, deep=True):
        """Get parameters of the estimator (sklearn-compatible)."""
        params = {}
        for key, value in self.__dict__.items():
            if deep and hasattr(value, "get_params"):
                deep_items = value.get_params().items()
                params.update({f"{key}__{k}": v for k, v in deep_items})
            else:
                params[key] = value
        return params

    def set_params(self, **params):
        """Set parameters of the estimator (sklearn-compatible)."""
        for key, value in params.items():
            if not hasattr(self, key):
                raise ValueError(f"Invalid parameter: {key}")
            setattr(self, key, value)
        return self

    def fit_predict(self, X):
        """Fit the filter and return state estimates."""
        return self.fit(X).predict()

    def score(self, X_true):
        """
        Negative MSE between filtered states and ground truth.

        Returns a negative value so that higher is better (sklearn convention).
        """
        self._check_fitted()
        X_true = np.asarray(X_true)
        if X_true.ndim == 1:
            X_true = X_true.reshape(-1, 1)
        states = self.filtered_states_
        if X_true.shape[1] != states.shape[1]:
            X_true = X_true[:, :states.shape[1]]
        return -np.mean((states - X_true[:len(states)]) ** 2)

    def validate_matrices(self, matrices):
        """Validate that matrices are NumPy or Dask arrays."""
        for name, matrix in matrices.items():
            if not isinstance(matrix, (np.ndarray, da.Array)):
                raise ValueError(f"{name} must be a NumPy or Dask array.")

    def _check_fitted(self):
        if not getattr(self, "is_fitted_", False):
            raise RuntimeError(
                f"{self.__class__.__name__} not fitted. Call fit() first."
            )

    def __repr__(self):
        return f"{self.__class__.__name__}()"
