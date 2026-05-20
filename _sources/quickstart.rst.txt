Quick Start
===========

All filters in TFiltersPy follow the same three-step workflow:

.. code-block:: text

   1. Create    -->   filter = KalmanFilter(F, H, Q, R, x0, P0)
   2. Fit       -->   filter.fit(measurements)
   3. Predict   -->   states = filter.predict()

Kalman Filter -- GPS Tracking
-----------------------------

A vehicle drives a figure-8 pattern. GPS gives noisy 2D position at 10 Hz.
We use a constant-velocity model to estimate the true trajectory.

**State vector:** ``[x, vx, y, vy]`` (position + velocity in 2D)

**Observation:** ``[x_gps, y_gps]`` (noisy GPS position)

.. code-block:: python

   import numpy as np
   from tfilterspy import KalmanFilter

   dt = 0.1  # 10 Hz GPS

   # Constant-velocity model
   F = np.array([[1, dt, 0,  0],
                 [0,  1, 0,  0],
                 [0,  0, 1, dt],
                 [0,  0, 0,  1]])

   H = np.array([[1, 0, 0, 0],   # GPS observes position only
                 [0, 0, 1, 0]])

   Q = np.diag([0.1, 1.0, 0.1, 1.0])   # process noise
   R = np.eye(2) * 25.0                  # GPS noise (5m std)^2
   x0 = np.zeros(4)
   P0 = np.eye(4) * 100.0

   kf = KalmanFilter(F, H, Q, R, x0, P0)
   kf.fit(gps_measurements)           # shape (n_steps, 2)
   filtered = kf.predict()            # shape (n_steps, 4)

   # RTS smoothing -- backward pass for even better estimates
   smoothed, smoothed_covs = kf.smooth()

   # Forecast 50 steps into the future
   forecast_states, forecast_covs = kf.forecast(50)

   # Evaluate against ground truth
   score = kf.score(true_states)      # negative MSE (higher = better)
   print(f"Log-likelihood: {kf.log_likelihood_:.1f}")

**Result:** 86% noise reduction (filtered), 96% (smoothed).

Extended Kalman Filter -- Radar Tracking
-----------------------------------------

An aircraft in a coordinated turn. A ground-based radar measures range and
bearing -- a nonlinear observation model. The EKF linearizes using Jacobians.

.. code-block:: python

   from tfilterspy import ExtendedKalmanFilter

   def f_aircraft(x):
       """Constant-velocity transition."""
       return np.array([x[0] + dt*x[1], x[1], x[2] + dt*x[3], x[3]])

   def h_radar(x):
       """Radar: [range, bearing] from origin."""
       r = np.sqrt(x[0]**2 + x[2]**2)
       theta = np.arctan2(x[2], x[0])
       return np.array([r, theta])

   def F_jac(x):
       return np.array([[1, dt, 0, 0], [0, 1, 0, 0],
                        [0, 0, 1, dt], [0, 0, 0, 1]])

   def H_jac(x):
       px, _, py, _ = x
       r = np.sqrt(px**2 + py**2) + 1e-10
       return np.array([[px/r, 0, py/r, 0],
                        [-py/r**2, 0, px/r**2, 0]])

   ekf = ExtendedKalmanFilter(
       f=f_aircraft, h=h_radar,
       F_jacobian=F_jac, H_jacobian=H_jac,
       Q=Q, R=R, x0=x0, P0=P0,
   )
   ekf.fit(radar_measurements)
   states = ekf.predict()
   smoothed, _ = ekf.smooth()   # RTS smoother works for EKF too

Unscented Kalman Filter -- No Jacobians
----------------------------------------

Same radar problem, but the UKF uses sigma points instead of Jacobians.
Often more accurate for highly nonlinear systems and easier to implement
since you only need the transition and observation functions.

.. code-block:: python

   from tfilterspy import UnscentedKalmanFilter

   ukf = UnscentedKalmanFilter(
       f=f_aircraft, h=h_radar,
       Q=Q, R=R, x0=x0, P0=P0,
       alpha=1e-3, beta=2.0, kappa=0.0,
   )
   ukf.fit(radar_measurements)
   states = ukf.predict()

Ensemble Kalman Filter -- High-Dimensional
--------------------------------------------

For systems with thousands of state variables (weather, ocean models) where
storing full covariance matrices is infeasible. Uses a Monte Carlo ensemble
with optional Dask parallelism.

.. code-block:: python

   from tfilterspy import EnsembleKalmanFilter

   enkf = EnsembleKalmanFilter(
       f=dynamics_fn, h=observation_fn,
       Q=Q, R=R, x0=x0,
       n_ensemble=100,
       use_dask=True,   # parallel propagation
   )
   enkf.fit(measurements)
   states = enkf.predict()

Particle Filter -- Robot Localization
--------------------------------------

A robot navigates using range measurements to known landmarks.
Handles nonlinear observations and multimodal distributions.

.. code-block:: python

   from tfilterspy import ParticleFilter

   def robot_dynamics(x):
       heading = x[2]
       return np.array([x[0] + 0.5*np.cos(heading),
                        x[1] + 0.5*np.sin(heading),
                        x[2] + 0.05])

   def range_to_landmarks(x):
       pos = x[:2]
       return np.sqrt(np.sum((landmarks - pos)**2, axis=1))

   pf = ParticleFilter(
       f=robot_dynamics, h=range_to_landmarks,
       Q=Q, R=R, x0=x0,
       n_particles=2000,
       resample_threshold=0.5,
   )
   pf.fit(range_measurements)
   states = pf.predict()
   print(f"Mean ESS: {np.mean(pf.effective_sample_sizes_):.0f}")

Online Filtering
-----------------

All filters support step-by-step updates for real-time / streaming applications:

.. code-block:: python

   kf = KalmanFilter(F, H, Q, R, x0, P0)
   for measurement in sensor_stream:
       x_est, P_est = kf.filter_step(measurement)
       # use x_est immediately for control, display, etc.
