Theory
======

This page covers the mathematical foundations of each filter implemented
in TFiltersPy.

Kalman Filter
-------------

The Kalman filter is optimal for **linear-Gaussian** state-space models:

.. math::

   x_{k+1} = F \, x_k + w_k, \quad w_k \sim \mathcal{N}(0, Q)

   z_k = H \, x_k + v_k, \quad v_k \sim \mathcal{N}(0, R)

**Predict step:**

.. math::

   \hat{x}_{k|k-1} = F \, \hat{x}_{k-1|k-1}

   P_{k|k-1} = F \, P_{k-1|k-1} \, F^\top + Q

**Update step** (Joseph form for numerical stability):

.. math::

   y_k = z_k - H \, \hat{x}_{k|k-1}

   S_k = H \, P_{k|k-1} \, H^\top + R

   K_k = P_{k|k-1} \, H^\top \, S_k^{-1}

   \hat{x}_{k|k} = \hat{x}_{k|k-1} + K_k \, y_k

   P_{k|k} = (I - K_k H) \, P_{k|k-1} \, (I - K_k H)^\top + K_k \, R \, K_k^\top

The Joseph form update for :math:`P_{k|k}` is more numerically stable than the
standard :math:`(I - K_k H) P_{k|k-1}` formulation.

**RTS Smoother** (backward pass):

.. math::

   C_k = P_{k|k} \, F^\top \, P_{k+1|k}^{-1}

   \hat{x}_{k|T} = \hat{x}_{k|k} + C_k (\hat{x}_{k+1|T} - \hat{x}_{k+1|k})

   P_{k|T} = P_{k|k} + C_k (P_{k+1|T} - P_{k+1|k}) C_k^\top

Extended Kalman Filter
-----------------------

For **nonlinear** systems with Gaussian noise:

.. math::

   x_{k+1} = f(x_k) + w_k, \quad z_k = h(x_k) + v_k

The EKF linearizes around the current estimate using first-order Taylor expansion:

.. math::

   F_k = \left.\frac{\partial f}{\partial x}\right|_{x=\hat{x}_{k|k}}, \quad
   H_k = \left.\frac{\partial h}{\partial x}\right|_{x=\hat{x}_{k|k-1}}

Then applies the standard Kalman equations with these linearized matrices.
Works well when the nonlinearity is mild relative to the uncertainty.

Unscented Kalman Filter
-------------------------

The UKF avoids Jacobians by propagating **sigma points** through the
nonlinear functions. For a state of dimension :math:`n`:

**Sigma point generation** (Merwe scaled):

.. math::

   \lambda = \alpha^2 (n + \kappa) - n

   \chi_0 = \hat{x}

   \chi_i = \hat{x} + \sqrt{(n + \lambda) P}_i, \quad i = 1, \ldots, n

   \chi_{n+i} = \hat{x} - \sqrt{(n + \lambda) P}_i, \quad i = 1, \ldots, n

**Weights:**

.. math::

   W_0^{(m)} = \frac{\lambda}{n + \lambda}, \quad
   W_0^{(c)} = \frac{\lambda}{n + \lambda} + (1 - \alpha^2 + \beta)

   W_i^{(m)} = W_i^{(c)} = \frac{1}{2(n + \lambda)}, \quad i = 1, \ldots, 2n

The sigma points are propagated through :math:`f` and :math:`h`, and the
resulting mean and covariance are computed as weighted averages. This captures
second-order statistics without derivatives.

Ensemble Kalman Filter
-----------------------

The EnKF approximates the Kalman filter using a **Monte Carlo ensemble**
of :math:`M` state realizations:

.. math::

   E_k = [x_k^{(1)}, x_k^{(2)}, \ldots, x_k^{(M)}]

**Predict:** Propagate each member through the dynamics:

.. math::

   x_{k+1}^{(i)} = f(x_k^{(i)}) + w_k^{(i)}

**Update:** Use ensemble statistics instead of full covariance:

.. math::

   \bar{x}_k = \frac{1}{M} \sum_{i=1}^{M} x_k^{(i)}

   P_k \approx \frac{1}{M-1} \sum_{i=1}^{M} (x_k^{(i)} - \bar{x}_k)(x_k^{(i)} - \bar{x}_k)^\top

The Kalman gain is computed from these ensemble-derived statistics.
Each member is updated with a perturbed observation:

.. math::

   x_k^{(i)} \leftarrow x_k^{(i)} + K_k(z_k + \epsilon_k^{(i)} - h(x_k^{(i)}))

This scales to systems with millions of state variables since we never
store or invert the full covariance matrix.

Particle Filter
----------------

The particle filter represents the posterior using :math:`N` weighted samples:

.. math::

   p(x_k | z_{1:k}) \approx \sum_{i=1}^{N} w_k^{(i)} \, \delta(x - x_k^{(i)})

**Predict:** Draw from the transition model:

.. math::

   x_k^{(i)} \sim f(x_{k-1}^{(i)}) + \mathcal{N}(0, Q)

**Update weights** using the observation likelihood:

.. math::

   w_k^{(i)} \propto w_{k-1}^{(i)} \cdot p(z_k | x_k^{(i)})

Weights are normalized to sum to 1.

**Systematic resampling** is triggered when the Effective Sample Size
drops below a threshold:

.. math::

   \text{ESS} = \frac{1}{\sum_{i=1}^{N} (w_k^{(i)})^2}

Systematic resampling has lower variance than multinomial resampling and
produces more evenly distributed samples.

**State estimate:**

.. math::

   \hat{x}_k = \sum_{i=1}^{N} w_k^{(i)} \, x_k^{(i)}

References
-----------

- R. E. Kalman, "A New Approach to Linear Filtering and Prediction Problems," *ASME Journal of Basic Engineering*, 1960.
- S. J. Julier and J. K. Uhlmann, "Unscented Filtering and Nonlinear Estimation," *Proceedings of the IEEE*, 2004.
- G. Evensen, "Sequential Data Assimilation with a Nonlinear Quasi-Geostrophic Model Using Monte Carlo Methods," *JGR*, 1994.
- A. Doucet, N. de Freitas, and N. Gordon, *Sequential Monte Carlo Methods in Practice*, Springer, 2001.
- H. E. Rauch, F. Tung, and C. T. Striebel, "Maximum Likelihood Estimates of Linear Dynamic Systems," *AIAA Journal*, 1965.
