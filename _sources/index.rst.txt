.. tfilters documentation master file

.. image:: _static/tfilters-logo.jpeg
   :alt: tfilters logo
   :width: 400px
   :align: center

.. raw:: html

   <br>

.. image:: https://img.shields.io/pypi/v/tfilterspy.svg
   :target: https://pypi.org/project/tfilterspy/
   :alt: PyPI version
.. image:: https://img.shields.io/pypi/pyversions/tfilterspy.svg
   :target: https://pypi.org/project/tfilterspy/
   :alt: Python versions
.. image:: https://img.shields.io/pypi/l/tfilterspy.svg
   :target: https://github.com/ubunye-ai-ecosystems/tfilterspy/blob/main/LICENSE
   :alt: License
.. image:: https://img.shields.io/github/stars/ubunye-ai-ecosystems/tfilterspy.svg?style=social
   :target: https://github.com/ubunye-ai-ecosystems/tfilterspy
   :alt: GitHub Stars

TFiltersPy
==========

**Bayesian state estimation for Python** -- sklearn-compatible, scalable, modular.

TFiltersPy provides 5 production-ready Bayesian filters with a unified
``fit`` / ``predict`` / ``score`` API. Built on NumPy/SciPy with optional
Dask parallelism for large-scale problems.

Part of the `Ubunye AI Ecosystems <https://github.com/ubunye-ai-ecosystems>`_.

Filters at a Glance
--------------------

.. list-table::
   :header-rows: 1
   :widths: 25 30 15 15 15

   * - Filter
     - Best For
     - Linearity
     - Jacobians?
     - Scales To
   * - **KalmanFilter**
     - GPS tracking, signal denoising
     - Linear
     - N/A
     - 10K+ steps
   * - **ExtendedKalmanFilter**
     - Radar, navigation
     - Nonlinear
     - Required
     - 10K+ steps
   * - **UnscentedKalmanFilter**
     - Highly nonlinear systems
     - Nonlinear
     - Not needed
     - 10K+ steps
   * - **EnsembleKalmanFilter**
     - Weather, ocean models
     - Nonlinear
     - Not needed
     - High-dim states
   * - **ParticleFilter**
     - Robot localization, multimodal
     - Any
     - Not needed
     - Non-Gaussian

Quick Example
-------------

.. code-block:: python

   import numpy as np
   from tfilterspy import KalmanFilter

   F = np.eye(2)
   H = np.eye(2)
   Q = np.eye(2) * 0.1
   R = np.eye(2) * 0.5

   kf = KalmanFilter(F, H, Q, R, x0=np.zeros(2), P0=np.eye(2))
   kf.fit(measurements)        # (n_steps, 2)
   filtered = kf.predict()     # (n_steps, 2)
   smoothed, _ = kf.smooth()   # RTS smoother

Key Features
-------------

- **5 Filters:** KF, EKF, UKF, EnKF, Particle Filter -- covering linear to fully nonlinear, Gaussian to arbitrary distributions
- **sklearn API:** ``fit()`` / ``predict()`` / ``score()`` / ``get_params()`` / ``set_params()``
- **RTS Smoothing:** Backward pass for KF and EKF to refine estimates
- **Online Filtering:** ``filter_step(z)`` for real-time streaming data
- **Forecasting:** ``forecast(n_steps)`` to predict future states
- **Dask Parallelism:** Ensemble and particle propagation on distributed clusters
- **Memory Efficient:** ``store_covariances=False`` saves ~80% memory on long series
- **Numerically Stable:** Joseph form covariance update, log-weight arithmetic

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   quickstart
   filter_guide
   examples
   api_reference
   literature
   modules
   CONTRIBUTING
   MAINTAINERS
