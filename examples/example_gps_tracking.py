"""
=============================================================
GPS Vehicle Tracking with Kalman Filter + RTS Smoother
=============================================================

A vehicle drives a figure-8 pattern. GPS measures its 2D position with
~5 m noise. We use a constant-velocity Kalman Filter to estimate
the true trajectory, then apply RTS smoothing for an even cleaner result.

Demonstrates:
- KalmanFilter: fit / predict / smooth / forecast / score
- How filtering reduces GPS noise by ~80%
- How smoothing further improves on filtering

State vector: [x, vx, y, vy]  (position + velocity in 2D)
Observation:  [x_gps, y_gps]  (noisy GPS position)
"""
import numpy as np

from tfilterspy import KalmanFilter


def generate_figure8_trajectory(n_steps, dt, speed=10.0, seed=42):
    """Generate a figure-8 ground truth + noisy GPS measurements."""
    rng = np.random.RandomState(seed)
    t = np.arange(n_steps) * dt
    omega = 2 * np.pi / (n_steps * dt / 2)

    x_true = speed * np.sin(omega * t)
    y_true = speed * np.sin(2 * omega * t) / 2
    vx_true = speed * omega * np.cos(omega * t)
    vy_true = speed * omega * np.cos(2 * omega * t)

    true_states = np.column_stack([x_true, vx_true, y_true, vy_true])

    gps_noise_std = 5.0  # meters
    gps_x = x_true + rng.randn(n_steps) * gps_noise_std
    gps_y = y_true + rng.randn(n_steps) * gps_noise_std
    measurements = np.column_stack([gps_x, gps_y])

    return true_states, measurements


def main():
    # --- Simulation parameters ---
    dt = 0.1        # 10 Hz GPS
    n_steps = 1000  # 100 seconds of driving

    # --- Generate data ---
    true_states, measurements = generate_figure8_trajectory(n_steps, dt)
    print(f"Generated {n_steps} timesteps of GPS data (dt={dt}s)")
    print(f"True state shape:  {true_states.shape}")
    print(f"Measurements shape: {measurements.shape}")

    # --- Build the constant-velocity model ---
    F = np.array([
        [1, dt, 0, 0],
        [0, 1,  0, 0],
        [0, 0,  1, dt],
        [0, 0,  0, 1],
    ])
    H = np.array([
        [1, 0, 0, 0],
        [0, 0, 1, 0],
    ])
    Q = np.diag([0.1, 1.0, 0.1, 1.0])   # process noise
    R = np.eye(2) * 25.0                  # GPS noise variance (5m std)^2
    x0 = np.array([0, 10, 0, 10], dtype=np.float64)
    P0 = np.eye(4) * 100.0

    # --- Create and run the Kalman Filter ---
    kf = KalmanFilter(F, H, Q, R, x0, P0)
    kf.fit(measurements)

    filtered = kf.predict()
    smoothed, _ = kf.smooth()

    # --- Forecast 50 steps into the future ---
    forecast_states, forecast_covs = kf.forecast(50)

    # --- Evaluate ---
    mse_gps = np.mean((measurements - true_states[:, [0, 2]]) ** 2)
    mse_filtered = np.mean((filtered[:, [0, 2]] - true_states[:, [0, 2]]) ** 2)
    mse_smoothed = np.mean((smoothed[:, [0, 2]] - true_states[:, [0, 2]]) ** 2)
    score = kf.score(true_states)

    print(f"\n{'='*50}")
    print(f"RESULTS")
    print(f"{'='*50}")
    print(f"MSE raw GPS:    {mse_gps:.3f} m^2")
    print(f"MSE filtered:   {mse_filtered:.3f} m^2  ({(1-mse_filtered/mse_gps)*100:.0f}% reduction)")
    print(f"MSE smoothed:   {mse_smoothed:.3f} m^2  ({(1-mse_smoothed/mse_gps)*100:.0f}% reduction)")
    print(f"Score (neg MSE): {score:.4f}")
    print(f"Log-likelihood:  {kf.log_likelihood_:.1f}")
    print(f"Forecast shape:  {forecast_states.shape}")

    # --- Online filtering demo ---
    print(f"\n{'='*50}")
    print("ONLINE FILTERING (filter_step)")
    print(f"{'='*50}")
    kf_online = KalmanFilter(F, H, Q, R, x0, P0)
    for i in range(5):
        x_est, P_est = kf_online.filter_step(measurements[i])
        print(f"  Step {i}: est=[{x_est[0]:.2f}, {x_est[2]:.2f}]  "
              f"true=[{true_states[i, 0]:.2f}, {true_states[i, 2]:.2f}]")

    # --- Plot if matplotlib available ---
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Trajectory plot
        ax = axes[0]
        ax.scatter(measurements[:, 0], measurements[:, 1],
                   s=1, alpha=0.3, c="gray", label="GPS")
        ax.plot(true_states[:, 0], true_states[:, 2],
                "k-", lw=2, label="True path")
        ax.plot(filtered[:, 0], filtered[:, 2],
                "b-", lw=1.5, label="Filtered")
        ax.plot(smoothed[:, 0], smoothed[:, 2],
                "r--", lw=1.5, label="Smoothed")
        ax.plot(forecast_states[:, 0], forecast_states[:, 2],
                "g:", lw=2, label="Forecast")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_title("GPS Vehicle Tracking — Figure-8")
        ax.legend()
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

        # Position error over time
        ax = axes[1]
        t = np.arange(n_steps) * dt
        err_gps = np.sqrt((measurements[:, 0] - true_states[:, 0]) ** 2 +
                          (measurements[:, 1] - true_states[:, 2]) ** 2)
        err_filt = np.sqrt((filtered[:, 0] - true_states[:, 0]) ** 2 +
                           (filtered[:, 2] - true_states[:, 2]) ** 2)
        err_smooth = np.sqrt((smoothed[:, 0] - true_states[:, 0]) ** 2 +
                             (smoothed[:, 2] - true_states[:, 2]) ** 2)

        ax.plot(t, err_gps, "gray", alpha=0.5, label=f"GPS (mean={err_gps.mean():.2f}m)")
        ax.plot(t, err_filt, "b", alpha=0.7, label=f"Filtered (mean={err_filt.mean():.2f}m)")
        ax.plot(t, err_smooth, "r", alpha=0.7, label=f"Smoothed (mean={err_smooth.mean():.2f}m)")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Position Error (m)")
        ax.set_title("Position Error Over Time")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("gps_tracking_result.png", dpi=150)
        plt.show()
        print("\nPlot saved to gps_tracking_result.png")
    except ImportError:
        print("\nInstall matplotlib to see plots: pip install matplotlib")


if __name__ == "__main__":
    main()
