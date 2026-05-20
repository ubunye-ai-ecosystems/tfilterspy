API Reference
=============

All filters inherit from ``BaseEstimator`` and share a common interface.

.. code-block:: text

                     +-----------------+
                     | BaseEstimator   |
                     |-----------------|
                     | fit(X)          |
                     | predict()       |
                     | score(X_true)   |
                     | fit_predict(X)  |
                     | get_params()    |
                     | set_params()    |
                     +--------+--------+
                              |
           +------------------+-------------------+
           |                  |                   |
   +-------v-------+  +------v--------+  +-------v--------+
   | KalmanFilter  |  | EKF/UKF/EnKF  |  | ParticleFilter |
   | smooth()      |  | (callables)   |  | ESS monitoring |
   | forecast()    |  |               |  | systematic     |
   +---------------+  +---------------+  | resampling     |
                                         +----------------+

Shared API (All Filters)
-------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Method
     - Description
   * - ``fit(X)``
     - Run forward filter on measurements ``X`` with shape ``(n_steps, n_obs)``.
       Returns ``self`` for chaining.
   * - ``predict()``
     - Return filtered state estimates as ndarray ``(n_steps, n_state)``.
   * - ``score(X_true)``
     - Negative MSE vs ground truth. Higher is better (sklearn convention).
   * - ``fit_predict(X)``
     - Shorthand for ``fit(X).predict()``.
   * - ``filter_step(z)``
     - Process a single measurement for online / streaming use.
       Returns ``(x_est, P_est)`` for Kalman-based, ``x_est`` for PF.
   * - ``get_params(deep=True)``
     - Return estimator parameters as dict (sklearn-compatible).
   * - ``set_params(**params)``
     - Set estimator parameters (sklearn-compatible).

KalmanFilter
-------------

.. code-block:: python

   from tfilterspy import KalmanFilter

   kf = KalmanFilter(F, H, Q, R, x0, P0, store_covariances=True)

**Constructor parameters:**

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Parameter
     - Description
   * - ``F`` : ndarray (n, n)
     - State transition matrix
   * - ``H`` : ndarray (m, n)
     - Observation matrix
   * - ``Q`` : ndarray (n, n)
     - Process noise covariance
   * - ``R`` : ndarray (m, m)
     - Observation noise covariance
   * - ``x0`` : ndarray (n,)
     - Initial state estimate
   * - ``P0`` : ndarray (n, n)
     - Initial state covariance
   * - ``store_covariances`` : bool
     - If ``False``, skip covariance storage to save ~80% memory.
       Disables ``smooth()``.

**Additional methods:**

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Method
     - Description
   * - ``smooth()``
     - RTS backward smoother. Returns ``(smoothed_states, smoothed_covs)``.
   * - ``forecast(n_steps)``
     - Predict ``n`` steps ahead. Returns ``(states, covariances)``.
   * - ``log_likelihood_``
     - Log-likelihood of the data (computed during ``fit``).

ExtendedKalmanFilter
---------------------

.. code-block:: python

   from tfilterspy import ExtendedKalmanFilter

   ekf = ExtendedKalmanFilter(
       f=transition_fn, h=observation_fn,
       F_jacobian=F_jac_fn, H_jacobian=H_jac_fn,
       Q=Q, R=R, x0=x0, P0=P0,
   )

**Constructor parameters:**

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Parameter
     - Description
   * - ``f`` : callable
     - State transition function ``f(x) -> x_next``
   * - ``h`` : callable
     - Observation function ``h(x) -> z``
   * - ``F_jacobian`` : callable
     - Jacobian of f: ``F_jacobian(x) -> ndarray (n, n)``
   * - ``H_jacobian`` : callable
     - Jacobian of h: ``H_jacobian(x) -> ndarray (m, n)``
   * - ``Q, R, x0, P0``
     - Same as KalmanFilter

**Additional methods:** ``smooth()`` (RTS smoother), ``log_likelihood_``

UnscentedKalmanFilter
----------------------

.. code-block:: python

   from tfilterspy import UnscentedKalmanFilter

   ukf = UnscentedKalmanFilter(
       f=transition_fn, h=observation_fn,
       Q=Q, R=R, x0=x0, P0=P0,
       alpha=1e-3, beta=2.0, kappa=0.0,
   )

**Sigma point parameters:**

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Parameter
     - Description
   * - ``alpha``
     - Spread of sigma points around the mean (default: 1e-3)
   * - ``beta``
     - Prior knowledge of distribution; 2.0 is optimal for Gaussian
   * - ``kappa``
     - Secondary scaling parameter (default: 0.0)

No Jacobians required -- just provide ``f(x)`` and ``h(x)``.

EnsembleKalmanFilter
---------------------

.. code-block:: python

   from tfilterspy import EnsembleKalmanFilter

   enkf = EnsembleKalmanFilter(
       f=transition_fn, h=observation_fn,
       Q=Q, R=R, x0=x0,
       n_ensemble=100,
       use_dask=True,
   )

**Constructor parameters:**

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Parameter
     - Description
   * - ``n_ensemble`` : int
     - Number of ensemble members (default: 50)
   * - ``use_dask`` : bool
     - Enable Dask parallel propagation (default: False)

Scales to high-dimensional state spaces where storing full covariance
matrices is infeasible.

ParticleFilter
---------------

.. code-block:: python

   from tfilterspy import ParticleFilter

   pf = ParticleFilter(
       f=transition_fn, h=observation_fn,
       Q=Q, R=R, x0=x0,
       n_particles=1000,
       resample_threshold=0.5,
       use_dask=False,
   )

**Constructor parameters:**

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Parameter
     - Description
   * - ``f`` : callable or ndarray
     - State transition. If ndarray, used as linear matrix (vectorized).
   * - ``h`` : callable or ndarray
     - Observation model. If ndarray, used as linear matrix (vectorized).
   * - ``n_particles`` : int
     - Number of particles (default: 1000)
   * - ``resample_threshold`` : float
     - Resample when ESS drops below this fraction of ``n_particles``
   * - ``use_dask`` : bool
     - Dask parallel particle propagation

**Attributes after fit:**

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Attribute
     - Description
   * - ``effective_sample_sizes_``
     - ESS at each time step (monitors particle degeneracy)
   * - ``log_likelihood_``
     - Log-likelihood of the data

Backward Compatibility
-----------------------

The legacy class names still work:

.. code-block:: python

   from tfilterspy import DaskKalmanFilter    # wraps KalmanFilter
   from tfilterspy import DaskParticleFilter  # wraps ParticleFilter

These accept the old constructor argument names and map them to the new API.
New code should use ``KalmanFilter`` and ``ParticleFilter`` directly.
