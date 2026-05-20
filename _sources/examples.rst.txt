Examples
========

TFiltersPy includes Python scripts and Jupyter notebooks demonstrating
each filter on real-world problems.

Python Examples
---------------

Run any example directly:

.. code-block:: bash

   python examples/example_gps_tracking.py

.. list-table::
   :header-rows: 1
   :widths: 30 15 55

   * - Example
     - Filter
     - Description
   * - ``example_gps_tracking.py``
     - KF
     - Vehicle tracks a figure-8 at 10 Hz GPS. Demonstrates ``fit``, ``predict``,
       ``smooth``, ``forecast``, ``score``, and ``filter_step``.
       Achieves 86% noise reduction (filtered), 96% (smoothed).
   * - ``example_radar_tracking.py``
     - EKF, UKF
     - Aircraft in a coordinated turn observed by radar (range + bearing).
       Compares EKF vs UKF accuracy. EKF smoothing reaches 81% improvement.
   * - ``example_robot_localization.py``
     - PF
     - Robot navigates using range measurements to 3 landmarks.
       Compares 100, 500, and 2000 particles. Shows ESS monitoring
       and online ``filter_step`` demo.

GPS Tracking (KalmanFilter)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   import numpy as np
   from tfilterspy import KalmanFilter

   dt = 0.1
   F = np.array([[1, dt, 0,  0],
                 [0,  1, 0,  0],
                 [0,  0, 1, dt],
                 [0,  0, 0,  1]])
   H = np.array([[1, 0, 0, 0],
                 [0, 0, 1, 0]])
   Q = np.diag([0.1, 1.0, 0.1, 1.0])
   R = np.eye(2) * 25.0
   x0 = np.array([0, 10, 0, 10], dtype=np.float64)
   P0 = np.eye(4) * 100.0

   kf = KalmanFilter(F, H, Q, R, x0, P0)
   kf.fit(measurements)
   filtered = kf.predict()
   smoothed, _ = kf.smooth()
   forecast_states, forecast_covs = kf.forecast(50)

.. code-block:: text

   RESULTS
   ==================================================
   MSE raw GPS:    24.465 m^2
   MSE filtered:   3.495 m^2  (86% reduction)
   MSE smoothed:   0.990 m^2  (96% reduction)
   Log-likelihood: -6193.8

Radar Tracking (EKF vs UKF)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from tfilterspy import ExtendedKalmanFilter, UnscentedKalmanFilter

   ekf = ExtendedKalmanFilter(
       f=f_aircraft, h=h_radar,
       F_jacobian=F_jac, H_jacobian=H_jac,
       Q=Q, R=R, x0=x0, P0=P0,
   )
   ekf.fit(measurements)
   ekf_states = ekf.predict()
   ekf_smoothed, _ = ekf.smooth()

   ukf = UnscentedKalmanFilter(
       f=f_aircraft, h=h_radar,
       Q=Q, R=R, x0=x0, P0=P0,
   )
   ukf.fit(measurements)
   ukf_states = ukf.predict()

.. code-block:: text

   POSITION RMSE COMPARISON
   =======================================================
   Raw radar:      42.84 m
   EKF filtered:   33.68 m  (21% improvement)
   EKF smoothed:    8.01 m  (81% improvement)
   UKF filtered:   33.66 m  (21% improvement)

Robot Localization (ParticleFilter)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   from tfilterspy import ParticleFilter

   pf = ParticleFilter(
       f=robot_dynamics, h=range_observation,
       Q=Q, R=R, x0=x0,
       n_particles=2000,
       resample_threshold=0.5,
   )
   pf.fit(measurements)
   states = pf.predict()
   print(f"Mean ESS: {np.mean(pf.effective_sample_sizes_):.0f}")

.. code-block:: text

   PARTICLE FILTER RESULTS
   ============================================================
   Particles    RMSE (m)    Mean ESS     ESS %
   -----------------------------------------------
        100       0.777       62.3     62.3%
        500       0.437      304.2     60.8%
       2000       0.320     1193.2     59.7%

Jupyter Notebooks
-----------------

Interactive notebooks with visualizations are in ``examples/notebooks/``.

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Notebook
     - Description
   * - ``Kalman-filters-use-cases.ipynb``
     - Image denoising (digits), EEG time series filtering,
       NLP topic smoothing with KalmanFilter
   * - ``particle-filters-usecases.ipynb``
     - Same three domains with ParticleFilter. Includes ESS monitoring
       and comparison to KF
   * - ``nonlinear-filters-usecases.ipynb``
     - Pendulum tracking (EKF vs UKF), radar with RTS smoothing,
       high-dimensional coupled oscillator (EnKF vs UKF)
   * - ``benchmarks.ipynb``
     - Speed and accuracy comparison of all 5 filters on the digits dataset.
       Includes classification accuracy with LogisticRegression

`Browse notebooks on GitHub <https://github.com/ubunye-ai-ecosystems/tfilterspy/tree/main/examples/notebooks>`_

Filter Pipeline Diagram
------------------------

.. code-block:: text

                                           +-----------+
   measurements ──> fit() ──> predict() ──>| filtered  |
                                  |        | states    |
                                  |        +-----------+
                                  |
                              smooth() ──> smoothed states (KF, EKF only)
                                  |
                            forecast(n) ──> future prediction (KF only)
                                  |
                           filter_step(z) ──> online / streaming mode (all filters)
