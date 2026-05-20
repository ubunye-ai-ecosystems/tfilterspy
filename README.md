<p align="center">
  <img src="branding/logo/tfilters-logo.jpeg?" alt="tfilterspy logo" width="300"/>
</p>

<h1 align="center">TFiltersPy</h1>

<p align="center">
  <strong>Bayesian state estimation for Python &mdash; sklearn-compatible, scalable, modular.</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/pypi/v/tfilterspy?color=blue&label=PyPI&style=for-the-badge" alt="PyPI Version"/>
  <img src="https://img.shields.io/pypi/pyversions/tfilterspy?style=for-the-badge" alt="Python Versions"/>
  <img src="https://img.shields.io/github/license/ubunye-ai-ecosystems/tfilterspy?color=green&style=for-the-badge" alt="License"/>
  <img src="https://img.shields.io/pypi/dm/tfilterspy?style=for-the-badge" alt="Downloads"/>
  <img src="https://img.shields.io/github/stars/ubunye-ai-ecosystems/tfilterspy?style=for-the-badge&logo=github" alt="Stars"/>
</p>

---

TFiltersPy provides **5 production-ready Bayesian filters** with a unified `fit` / `predict` / `score` API. Built on NumPy/SciPy with optional Dask parallelism for large-scale problems.

```
pip install tfilterspy
```

## Filters at a Glance

| Filter | Best For | Linearity | Jacobians? | Scales To |
|--------|----------|-----------|------------|-----------|
| **KalmanFilter** | GPS tracking, signal denoising | Linear | N/A | 10K+ steps |
| **ExtendedKalmanFilter** | Radar, navigation | Nonlinear | Required | 10K+ steps |
| **UnscentedKalmanFilter** | Highly nonlinear systems | Nonlinear | Not needed | 10K+ steps |
| **EnsembleKalmanFilter** | Weather, ocean models | Nonlinear | Not needed | High-dim states |
| **ParticleFilter** | Robot localization, multimodal | Any | Not needed | Non-Gaussian |

## Architecture

```
tfilterspy/
  __init__.py               # Top-level exports
  base_estimator.py          # BaseEstimator (sklearn API: fit/predict/score)
  state_estimation/
    linear_filters.py        # KalmanFilter, DaskKalmanFilter
    extended_kalman.py       # ExtendedKalmanFilter
    unscented_kalman.py      # UnscentedKalmanFilter
    ensemble_kalman.py       # EnsembleKalmanFilter (Dask-parallel)
    particle_filters.py      # ParticleFilter, DaskParticleFilter
  utils/
    optimisation_utils.py    # ParameterEstimator
```

```
                    +-----------------+
                    | BaseEstimator   |
                    |-----------------|
                    | fit(X)          |
                    | predict()       |
                    | score(X_true)   |
                    | fit_predict(X)  |
                    | filter_step(z)  |
                    | get_params()    |
                    | set_params()    |
                    +--------+--------+
                             |
          +------------------+-------------------+
          |                  |                   |
  +-------v-------+  +------v--------+  +-------v--------+
  | KalmanFilter  |  | Nonlinear     |  | ParticleFilter |
  |               |  | Filters       |  |                |
  | F, H matrices |  | f(), h()      |  | f(), h() or    |
  | smooth()      |  | callables     |  | F, H matrices  |
  | forecast()    |  |               |  |                |
  +---------------+  +---+-------+---+  | n_particles    |
                         |       |      | ESS monitoring |
              +----------+   +---+---+  +----------------+
              |              |       |
        +-----v-----+  +----v--+ +--v--------+
        |    EKF    |  | UKF   | |   EnKF    |
        | Jacobians |  | Sigma | | Ensemble  |
        | F_jac,    |  | points| | Dask      |
        | H_jac     |  |       | | parallel  |
        +-----------+  +-------+ +-----------+
```

## Quick Start

### Kalman Filter &mdash; GPS Tracking

A vehicle drives a figure-8. GPS gives noisy 2D position. The Kalman filter estimates the true trajectory, and RTS smoothing refines it further.

```python
import numpy as np
from tfilterspy import KalmanFilter

dt = 0.1  # 10 Hz GPS

# Constant-velocity model: state = [x, vx, y, vy]
F = np.array([[1, dt, 0,  0],
              [0,  1, 0,  0],
              [0,  0, 1, dt],
              [0,  0, 0,  1]])

H = np.array([[1, 0, 0, 0],    # GPS observes position only
              [0, 0, 1, 0]])

Q = np.diag([0.1, 1.0, 0.1, 1.0])   # process noise
R = np.eye(2) * 25.0                  # GPS noise (5m std)^2
x0 = np.zeros(4)
P0 = np.eye(4) * 100.0

kf = KalmanFilter(F, H, Q, R, x0, P0)
kf.fit(gps_measurements)           # shape (n_steps, 2)
filtered = kf.predict()            # shape (n_steps, 4)
smoothed, _ = kf.smooth()          # RTS smoother
forecast, _ = kf.forecast(50)      # 50 steps ahead
score = kf.score(true_states)      # negative MSE
```

```
Filter Pipeline:
                                          +----------+
  GPS measurements ──> fit() ──> predict() ──> filtered states
                                     |
                                  smooth() ──> smoothed states (better)
                                     |
                                 forecast(n) ──> future prediction
```

### Extended Kalman Filter &mdash; Radar Tracking

An aircraft in a coordinated turn. Radar measures range and bearing (nonlinear). The EKF linearizes at each step using Jacobians.

```python
from tfilterspy import ExtendedKalmanFilter

def f_aircraft(x):
    return np.array([x[0] + dt*x[1], x[1], x[2] + dt*x[3], x[3]])

def h_radar(x):
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
smoothed, _ = ekf.smooth()
```

### Unscented Kalman Filter &mdash; No Jacobians Needed

Same radar problem, but the UKF uses sigma points instead of Jacobians. Often more accurate for highly nonlinear systems.

```python
from tfilterspy import UnscentedKalmanFilter

ukf = UnscentedKalmanFilter(
    f=f_aircraft, h=h_radar,
    Q=Q, R=R, x0=x0, P0=P0,
    alpha=1e-3, beta=2.0, kappa=0.0,
)
ukf.fit(radar_measurements)
states = ukf.predict()
```

### Ensemble Kalman Filter &mdash; High-Dimensional Systems

Weather models, ocean simulations, and other systems with thousands of state variables. Uses a Monte Carlo ensemble instead of full covariance matrices. Supports Dask for parallel ensemble propagation.

```python
from tfilterspy import EnsembleKalmanFilter

enkf = EnsembleKalmanFilter(
    f=dynamics_fn, h=observation_fn,
    Q=Q, R=R, x0=x0,
    n_ensemble=100,
    use_dask=True,      # parallel ensemble propagation
)
enkf.fit(measurements)
states = enkf.predict()
```

### Particle Filter &mdash; Robot Localization

A robot navigates using range measurements to known landmarks. The particle filter handles the nonlinear observation model and can represent multimodal distributions.

```python
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
ess = pf.effective_sample_sizes_   # monitor particle health
```

## Online Filtering (Streaming Data)

All filters support step-by-step updates for real-time applications:

```python
kf = KalmanFilter(F, H, Q, R, x0, P0)
for measurement in sensor_stream:
    x_est, P_est = kf.filter_step(measurement)
    print(f"State estimate: {x_est}")
```

## Choosing a Filter

```
Start
  |
  v
Is the system linear? ──yes──> KalmanFilter
  |                              (fastest, optimal for linear-Gaussian)
  no
  |
  v
Is noise Gaussian? ──no──> ParticleFilter
  |                          (handles any distribution)
  yes
  |
  v
Can you compute Jacobians? ──yes──> ExtendedKalmanFilter
  |                                   (fast, needs analytical derivatives)
  no
  |
  v
Is the state dimension large (>100)? ──yes──> EnsembleKalmanFilter
  |                                             (scales to millions of states)
  no
  |
  v
UnscentedKalmanFilter
  (no Jacobians, sigma-point accuracy)
```

## Memory Efficiency

For long time series, disable covariance storage to save ~80% memory:

```python
kf = KalmanFilter(F, H, Q, R, x0, P0, store_covariances=False)
kf.fit(million_step_data)     # only stores state estimates
states = kf.predict()         # works fine
# kf.smooth()                 # raises error (needs stored covariances)
```

## Examples

Full working examples are included in the repository:

| Example | File | Description |
|---------|------|-------------|
| GPS Vehicle Tracking | [`examples/example_gps_tracking.py`](examples/example_gps_tracking.py) | KF + RTS smoother on figure-8 trajectory, 86% noise reduction |
| Radar Target Tracking | [`examples/example_radar_tracking.py`](examples/example_radar_tracking.py) | EKF vs UKF on coordinated-turn aircraft with polar radar |
| Robot Localization | [`examples/example_robot_localization.py`](examples/example_robot_localization.py) | Particle filter with range measurements to landmarks |

### Notebooks

| Notebook | Description |
|----------|-------------|
| [`Kalman-filters-use-cases.ipynb`](examples/notebooks/Kalman-filters-use-cases.ipynb) | Image denoising, EEG filtering, NLP topic smoothing |
| [`particle-filters-usecases.ipynb`](examples/notebooks/particle-filters-usecases.ipynb) | Computer vision, EEG, topic tracking with particle filters |
| [`nonlinear-filters-usecases.ipynb`](examples/notebooks/nonlinear-filters-usecases.ipynb) | EKF, UKF, and EnKF on nonlinear problems |
| [`benchmarks.ipynb`](examples/notebooks/benchmarks.ipynb) | Speed and accuracy comparison of all 5 filters |

Run any example:

```bash
python examples/example_gps_tracking.py
```

## API Reference

All filters inherit from `BaseEstimator` and share this interface:

| Method | Description |
|--------|-------------|
| `fit(X)` | Run the forward filter pass on measurements `X` (n_steps, n_obs) |
| `predict()` | Return filtered state estimates (n_steps, n_state) |
| `score(X_true)` | Negative MSE vs ground truth (higher = better, sklearn convention) |
| `fit_predict(X)` | Shorthand for `fit(X).predict()` |
| `filter_step(z)` | Process a single measurement for online/streaming use |
| `get_params()` | Get estimator parameters (sklearn-compatible) |
| `set_params(**p)` | Set estimator parameters (sklearn-compatible) |

**KalmanFilter** also provides:

| Method | Description |
|--------|-------------|
| `smooth()` | RTS backward smoother, returns `(smoothed_states, smoothed_covs)` |
| `forecast(n_steps)` | Predict `n` steps into the future, returns `(states, covariances)` |

**ParticleFilter** also provides:

| Attribute | Description |
|-----------|-------------|
| `effective_sample_sizes_` | ESS at each time step (monitors particle degeneracy) |

## Installation

```bash
pip install tfilterspy
```

From source:

```bash
git clone https://github.com/ubunye-ai-ecosystems/tfilterspy.git
cd tfilterspy
pip install -e .[dev]
```

**Requirements:** Python >= 3.8, NumPy, SciPy, Dask

## Development

```bash
git clone https://github.com/ubunye-ai-ecosystems/tfilterspy.git
cd tfilterspy
pip install -e .[dev]
pytest tests/
```

The test suite covers all 5 filters with 37+ tests including edge cases, backward compatibility, and a 10K-step large dataset test.

## Contributing

- Found a bug? Open an [issue](https://github.com/ubunye-ai-ecosystems/tfilterspy/issues).
- Want to add a feature? Fork, implement, and open a pull request.
- Write tests for any new filter or feature.

## License

MIT License. See [LICENSE](LICENSE) for details.

## Documentation

Full documentation: [ubunye-ai-ecosystems.github.io/tfilterspy](https://ubunye-ai-ecosystems.github.io/tfilterspy)
