"""
=============================================================
Robot Localization with Particle Filter
=============================================================

A mobile robot moves through a 2D environment with known landmarks.
The robot measures its distance to each landmark — a nonlinear
observation model with potential multimodality.

This is a classic problem where particle filters excel because:
- The observation model is nonlinear (range measurements)
- The posterior can be multimodal (ambiguous landmarks)
- The process noise may be non-Gaussian

Demonstrates:
- ParticleFilter: fit / predict / filter_step
- Vectorized matrix-based propagation vs callable observation model
- Effective sample size monitoring

State vector:  [x, y, heading]  (robot pose)
Observation:   [d1, d2, d3]    (range to 3 landmarks)
"""
import numpy as np

from tfilterspy import ParticleFilter


# --- Landmark positions ---
LANDMARKS = np.array([
    [0.0, 10.0],
    [10.0, 0.0],
    [10.0, 10.0],
])
N_LANDMARKS = len(LANDMARKS)


def robot_dynamics(x):
    """Simple unicycle model: move forward + rotate."""
    speed = 0.5
    turn_rate = 0.05
    heading = x[2]
    return np.array([
        x[0] + speed * np.cos(heading),
        x[1] + speed * np.sin(heading),
        x[2] + turn_rate,
    ])


def range_observation(x):
    """Measure range to each landmark."""
    pos = x[:2]
    ranges = np.sqrt(np.sum((LANDMARKS - pos) ** 2, axis=1))
    return ranges


def generate_robot_data(n_steps, seed=42):
    """Generate robot trajectory and noisy range measurements."""
    rng = np.random.RandomState(seed)

    true_states = np.zeros((n_steps, 3))
    x = np.array([0.0, 0.0, np.pi / 4])  # start position

    range_noise_std = 0.5  # meters
    process_noise_std = np.array([0.1, 0.1, 0.02])

    measurements = np.zeros((n_steps, N_LANDMARKS))

    for i in range(n_steps):
        x = robot_dynamics(x) + rng.randn(3) * process_noise_std
        true_states[i] = x
        measurements[i] = range_observation(x) + rng.randn(N_LANDMARKS) * range_noise_std

    return true_states, measurements


def main():
    n_steps = 200
    true_states, measurements = generate_robot_data(n_steps)

    print(f"Robot localization: {n_steps} timesteps")
    print(f"Landmarks: {LANDMARKS.tolist()}")
    print(f"Start: ({true_states[0, 0]:.2f}, {true_states[0, 1]:.2f})")
    print(f"End:   ({true_states[-1, 0]:.2f}, {true_states[-1, 1]:.2f})")

    # --- Build Particle Filter ---
    Q = np.diag([0.1, 0.1, 0.02]) ** 2  # process noise covariance
    R = np.eye(N_LANDMARKS) * 0.5 ** 2   # range noise covariance
    x0 = np.array([0.0, 0.0, np.pi / 4])

    n_particles_list = [100, 500, 2000]
    results = {}

    for n_particles in n_particles_list:
        pf = ParticleFilter(
            f=robot_dynamics,
            h=range_observation,
            Q=Q,
            R=R,
            x0=x0,
            n_particles=n_particles,
            resample_threshold=0.5,
        )
        pf.fit(measurements)
        states = pf.predict()

        pos_rmse = np.sqrt(np.mean(
            (states[:, 0] - true_states[:, 0]) ** 2 +
            (states[:, 1] - true_states[:, 1]) ** 2
        ))
        mean_ess = np.mean(pf.effective_sample_sizes_)
        results[n_particles] = (states, pos_rmse, mean_ess, pf)

    # --- Print results ---
    print(f"\n{'='*60}")
    print(f"PARTICLE FILTER RESULTS")
    print(f"{'='*60}")
    print(f"{'Particles':>10}  {'RMSE (m)':>10}  {'Mean ESS':>10}  {'ESS %':>8}")
    print(f"{'-'*45}")
    for np_ in n_particles_list:
        _, rmse, ess, _ = results[np_]
        print(f"{np_:>10d}  {rmse:>10.3f}  {ess:>10.1f}  {ess/np_*100:>7.1f}%")

    # --- Online filtering demo ---
    print(f"\n{'='*60}")
    print("ONLINE FILTERING (filter_step)")
    print(f"{'='*60}")
    pf_online = ParticleFilter(
        f=robot_dynamics, h=range_observation,
        Q=Q, R=R, x0=x0, n_particles=1000,
    )
    for i in range(10):
        state = pf_online.filter_step(measurements[i])
        err = np.sqrt((state[0] - true_states[i, 0]) ** 2 +
                      (state[1] - true_states[i, 1]) ** 2)
        print(f"  Step {i:2d}: est=({state[0]:6.2f}, {state[1]:6.2f})  "
              f"true=({true_states[i, 0]:6.2f}, {true_states[i, 1]:6.2f})  "
              f"err={err:.2f}m")

    # --- Score ---
    best_pf = results[2000][3]
    score = best_pf.score(true_states)
    print(f"\nScore (neg MSE, 2000 particles): {score:.4f}")

    # --- Plot ---
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # Trajectory comparison
        ax = axes[0]
        ax.plot(true_states[:, 0], true_states[:, 1],
                "k-", lw=2, label="True path")
        colors = ["blue", "orange", "red"]
        for (np_, color) in zip(n_particles_list, colors):
            states, rmse, _, _ = results[np_]
            ax.plot(states[:, 0], states[:, 1],
                    "-", color=color, alpha=0.7,
                    label=f"PF {np_} ({rmse:.2f}m)")
        ax.scatter(LANDMARKS[:, 0], LANDMARKS[:, 1],
                   s=200, marker="^", c="green", zorder=5, label="Landmarks")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_title("Robot Localization")
        ax.legend(fontsize=9)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

        # Position error over time
        ax = axes[1]
        t = np.arange(n_steps)
        for (np_, color) in zip(n_particles_list, colors):
            states, _, _, _ = results[np_]
            err = np.sqrt((states[:, 0] - true_states[:, 0]) ** 2 +
                          (states[:, 1] - true_states[:, 1]) ** 2)
            ax.plot(t, err, color=color, alpha=0.7, label=f"{np_} particles")
        ax.set_xlabel("Time step")
        ax.set_ylabel("Position Error (m)")
        ax.set_title("Error Over Time")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # ESS over time
        ax = axes[2]
        for (np_, color) in zip(n_particles_list, colors):
            _, _, _, pf = results[np_]
            ax.plot(t, pf.effective_sample_sizes_ / np_ * 100,
                    color=color, alpha=0.7, label=f"{np_} particles")
        ax.axhline(y=50, color="gray", linestyle="--", alpha=0.5, label="Resample threshold")
        ax.set_xlabel("Time step")
        ax.set_ylabel("ESS (%)")
        ax.set_title("Effective Sample Size")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("robot_localization_result.png", dpi=150)
        plt.show()
        print("\nPlot saved to robot_localization_result.png")
    except ImportError:
        print("\nInstall matplotlib to see plots: pip install matplotlib")


if __name__ == "__main__":
    main()
