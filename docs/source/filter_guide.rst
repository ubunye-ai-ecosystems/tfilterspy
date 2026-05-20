Choosing a Filter
=================

Use this guide to pick the right filter for your problem.

Decision Flowchart
------------------

.. code-block:: text

   Is the system linear?
     |
     +-- YES --> KalmanFilter
     |             (fastest, optimal for linear-Gaussian)
     |
     +-- NO
          |
          Is the noise Gaussian?
            |
            +-- NO --> ParticleFilter
            |            (handles any distribution, multimodal)
            |
            +-- YES
                 |
                 Can you compute Jacobians?
                   |
                   +-- YES --> ExtendedKalmanFilter
                   |             (fast, needs analytical derivatives)
                   |
                   +-- NO
                        |
                        Is the state dimension large (>100)?
                          |
                          +-- YES --> EnsembleKalmanFilter
                          |             (scales to millions of states, Dask-parallel)
                          |
                          +-- NO --> UnscentedKalmanFilter
                                       (sigma-point accuracy, no Jacobians)

Filter Comparison
------------------

.. list-table::
   :header-rows: 1
   :widths: 20 15 15 15 15 20

   * - Property
     - KF
     - EKF
     - UKF
     - EnKF
     - PF
   * - Linearity
     - Linear
     - Nonlinear
     - Nonlinear
     - Nonlinear
     - Any
   * - Noise assumption
     - Gaussian
     - Gaussian
     - Gaussian
     - Gaussian
     - Any
   * - Jacobians needed
     - No
     - Yes
     - No
     - No
     - No
   * - Multimodal support
     - No
     - No
     - No
     - No
     - Yes
   * - Smoothing
     - RTS
     - RTS
     - --
     - --
     - --
   * - Forecasting
     - Yes
     - --
     - --
     - --
     - --
   * - Dask parallel
     - --
     - --
     - --
     - Yes
     - Yes
   * - Computation cost
     - O(n^3)
     - O(n^3)
     - O(n^3)
     - O(n^2 * M)
     - O(n * N)
   * - Best accuracy
     - Linear systems
     - Mild nonlinearity
     - Strong nonlinearity
     - High-dim
     - Multimodal

Where ``n`` = state dimension, ``M`` = ensemble size, ``N`` = particle count.

When to Use Each Filter
------------------------

**KalmanFilter**

- GPS/INS navigation with linear dynamics
- Signal denoising (e.g., sensor data, images, EEG)
- Financial time series smoothing
- Any system where ``x_{k+1} = F @ x_k + noise``

**ExtendedKalmanFilter**

- Radar tracking (range + bearing observations)
- Satellite orbit determination
- SLAM (simultaneous localization and mapping)
- Systems where you can derive Jacobians analytically

**UnscentedKalmanFilter**

- Same problems as EKF but without needing Jacobians
- Strongly nonlinear dynamics where EKF linearization is poor
- Faster prototyping (just provide ``f(x)`` and ``h(x)``)

**EnsembleKalmanFilter**

- Weather prediction / data assimilation
- Ocean current modeling
- Any system with state dimension > 100 where full covariance is infeasible
- Set ``use_dask=True`` for parallel ensemble propagation

**ParticleFilter**

- Robot localization with landmark observations
- Target tracking with clutter
- Multimodal posterior distributions
- Non-Gaussian noise (e.g., heavy-tailed, uniform)

Memory Efficiency
------------------

For long time series, disable covariance storage:

.. code-block:: python

   kf = KalmanFilter(F, H, Q, R, x0, P0, store_covariances=False)
   kf.fit(million_step_data)     # only stores state estimates
   states = kf.predict()         # works
   # kf.smooth()                 # raises error (needs stored covariances)

This saves approximately 80% of memory for the KalmanFilter on long sequences.
