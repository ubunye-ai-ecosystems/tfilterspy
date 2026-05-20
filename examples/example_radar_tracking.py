"""
=============================================================
Radar Target Tracking — EKF vs UKF Comparison
=============================================================

An aircraft flies a curved trajectory. A ground-based radar measures
range and bearing (polar coordinates) — a nonlinear observation model.

This example compares:
- ExtendedKalmanFilter (EKF): uses analytical Jacobians
- UnscentedKalmanFilter (UKF): uses sigma points, no Jacobians

State vector: [x, vx, y, vy]  (position + velocity in 2D)
Observation:  [range, bearing]  (polar from origin)

    range   = sqrt(x^2 + y^2)
    bearing = atan2(y, x)
"""
import numpy as np

from tfilterspy import ExtendedKalmanFilter, UnscentedKalmanFilter


def generate_aircraft_data(n_steps, dt, seed=42):
    """Aircraft in a coordinated turn (constant angular rate)."""
    rng = np.random.RandomState(seed)
    omega = 0.02  # rad/s turn rate
    speed = 100.0  # m/s

    true_states = np.zeros((n_steps, 4))
    x, vx, y, vy = 1000.0, speed, 500.0, 0.0

    for i in range(n_steps):
        true_states[i] = [x, vx, y, vy]
        # Coordinated turn dynamics
        x += vx * dt
        y += vy * dt
        new_vx = vx * np.cos(omega * dt) - vy * np.sin(omega * dt)
        new_vy = vx * np.sin(omega * dt) + vy * np.cos(omega * dt)
        vx, vy = new_vx, new_vy
        # Add process noise
        x += rng.randn() * 1.0
        y += rng.randn() * 1.0

    # Generate radar measurements
    range_noise_std = 50.0    # meters
    bearing_noise_std = 0.02  # radians (~1 degree)

    ranges = np.sqrt(true_states[:, 0] ** 2 + true_states[:, 2] ** 2)
    bearings = np.arctan2(true_states[:, 2], true_states[:, 0])

    measurements = np.column_stack([
        ranges + rng.randn(n_steps) * range_noise_std,
        bearings + rng.randn(n_steps) * bearing_noise_std,
    ])

    return true_states, measurements


# --- System model ---
DT = 0.5  # radar scan interval


def f_aircraft(x):
    """Constant-velocity state transition."""
    return np.array([
        x[0] + DT * x[1],
        x[1],
        x[2] + DT * x[3],
        x[3],
    ])


def h_radar(x):
    """Radar observation: [range, bearing]."""
    r = np.sqrt(x[0] ** 2 + x[2] ** 2)
    theta = np.arctan2(x[2], x[0])
    return np.array([r, theta])


def F_jacobian(x):
    """Jacobian of f (linear, so constant)."""
    return np.array([
        [1, DT, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, DT],
        [0, 0, 0, 1],
    ])


def H_jacobian(x):
    """Jacobian of the radar observation model."""
    px, _, py, _ = x
    r = np.sqrt(px ** 2 + py ** 2) + 1e-10
    r2 = r ** 2
    return np.array([
        [px / r, 0, py / r, 0],
        [-py / r2, 0, px / r2, 0],
    ])


def main():
    n_steps = 500
    true_states, measurements = generate_aircraft_data(n_steps, DT)
    print(f"Radar tracking: {n_steps} scans, dt={DT}s")
    print(f"Aircraft starts at ({true_states[0, 0]:.0f}, {true_states[0, 2]:.0f})")

    # Common parameters
    Q = np.diag([1.0, 0.5, 1.0, 0.5])
    R = np.diag([50.0 ** 2, 0.02 ** 2])  # range and bearing noise
    x0 = np.array([1000.0, 100.0, 500.0, 0.0])
    P0 = np.diag([100.0, 10.0, 100.0, 10.0])

    # --- EKF ---
    ekf = ExtendedKalmanFilter(
        f=f_aircraft, h=h_radar,
        F_jacobian=F_jacobian, H_jacobian=H_jacobian,
        Q=Q, R=R, x0=x0, P0=P0,
    )
    ekf.fit(measurements)
    ekf_states = ekf.predict()
    ekf_smoothed, _ = ekf.smooth()

    # --- UKF ---
    ukf = UnscentedKalmanFilter(
        f=f_aircraft, h=h_radar,
        Q=Q, R=R, x0=x0, P0=P0,
        alpha=1e-3, beta=2.0, kappa=0.0,
    )
    ukf.fit(measurements)
    ukf_states = ukf.predict()

    # --- Evaluate ---
    def position_rmse(est, true):
        return np.sqrt(np.mean((est[:, 0] - true[:, 0]) ** 2 +
                               (est[:, 2] - true[:, 2]) ** 2))

    # Convert measurements back to Cartesian for comparison
    meas_x = measurements[:, 0] * np.cos(measurements[:, 1])
    meas_y = measurements[:, 0] * np.sin(measurements[:, 1])

    rmse_radar = np.sqrt(np.mean((meas_x - true_states[:, 0]) ** 2 +
                                 (meas_y - true_states[:, 2]) ** 2))
    rmse_ekf = position_rmse(ekf_states, true_states)
    rmse_ekf_smooth = position_rmse(ekf_smoothed, true_states)
    rmse_ukf = position_rmse(ukf_states, true_states)

    print(f"\n{'='*55}")
    print(f"POSITION RMSE COMPARISON")
    print(f"{'='*55}")
    print(f"Raw radar:     {rmse_radar:8.2f} m")
    print(f"EKF filtered:  {rmse_ekf:8.2f} m  ({(1-rmse_ekf/rmse_radar)*100:.0f}% improvement)")
    print(f"EKF smoothed:  {rmse_ekf_smooth:8.2f} m  ({(1-rmse_ekf_smooth/rmse_radar)*100:.0f}% improvement)")
    print(f"UKF filtered:  {rmse_ukf:8.2f} m  ({(1-rmse_ukf/rmse_radar)*100:.0f}% improvement)")
    print(f"\nEKF log-likelihood: {ekf.log_likelihood_:.1f}")
    print(f"UKF log-likelihood: {ukf.log_likelihood_:.1f}")

    winner = "UKF" if rmse_ukf < rmse_ekf else "EKF"
    print(f"\nWinner: {winner} (by {abs(rmse_ekf - rmse_ukf):.2f} m)")

    # --- Plot ---
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Trajectory
        ax = axes[0]
        ax.scatter(meas_x, meas_y, s=1, alpha=0.2, c="gray", label="Radar")
        ax.plot(true_states[:, 0], true_states[:, 2],
                "k-", lw=2, label="True")
        ax.plot(ekf_states[:, 0], ekf_states[:, 2],
                "b-", lw=1.5, alpha=0.8, label=f"EKF ({rmse_ekf:.1f}m)")
        ax.plot(ukf_states[:, 0], ukf_states[:, 2],
                "r--", lw=1.5, alpha=0.8, label=f"UKF ({rmse_ukf:.1f}m)")
        ax.plot(ekf_smoothed[:, 0], ekf_smoothed[:, 2],
                "g:", lw=1.5, alpha=0.8, label=f"EKF smooth ({rmse_ekf_smooth:.1f}m)")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_title("Radar Target Tracking — Coordinated Turn")
        ax.legend(fontsize=9)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

        # Error over time
        ax = axes[1]
        t = np.arange(n_steps) * DT
        err_ekf = np.sqrt((ekf_states[:, 0] - true_states[:, 0]) ** 2 +
                          (ekf_states[:, 2] - true_states[:, 2]) ** 2)
        err_ukf = np.sqrt((ukf_states[:, 0] - true_states[:, 0]) ** 2 +
                          (ukf_states[:, 2] - true_states[:, 2]) ** 2)
        ax.plot(t, err_ekf, "b", alpha=0.7, label="EKF")
        ax.plot(t, err_ukf, "r", alpha=0.7, label="UKF")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Position Error (m)")
        ax.set_title("EKF vs UKF Error Over Time")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("radar_tracking_result.png", dpi=150)
        plt.show()
        print("\nPlot saved to radar_tracking_result.png")
    except ImportError:
        print("\nInstall matplotlib to see plots: pip install matplotlib")


if __name__ == "__main__":
    main()
