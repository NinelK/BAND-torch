import argparse
import os

import numpy as np
import torch
from scipy.integrate import odeint


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def lorenz_dynamics(state, t, sigma=10.0, rho=28.0, beta=8.0 / 3.0):
    x, y, z = state
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z
    return [dxdt, dydt, dzdt]


def generate_lorenz_dataset(base_dir, bin_sz_ms=20):
    dataset_name = "Synthetic_Lorenz_Offmanifold_Unidir_Kicks"
    print(f"Generating {dataset_name} (bin_sz={bin_sz_ms}ms)...")
    out_dir = os.path.join(base_dir, f"data_{dataset_name}")
    os.makedirs(out_dir, exist_ok=True)

    n_neurons = 200
    n_latents_true = 3
    n_bins = 100
    dt = bin_sz_ms / 1000.0

    n_img_train = 1600
    n_img_valid = 400
    n_img_test = 400

    print("Initializing Global Readout Matrices...")
    C = 1.0 * torch.randn(n_neurons, n_latents_true)
    d = 0.0 + 0.2 * torch.randn(n_neurons)
    C_vel = torch.eye(n_latents_true)  # torch.randn(2, n_latents_true)
    B_in = torch.eye(n_latents_true)

    # --- Sample a random unit vector ONCE for the whole dataset ---
    print("Sampling Fixed Unidirectional Kick Vector...")
    direction_vec = np.random.randn(n_latents_true)
    direction_vec = direction_vec / np.linalg.norm(direction_vec)
    print(f"Sampled Direction: {direction_vec}")

    # --- Sample another random unit vector for off-manifold input footprint ---
    gen_input_vec = torch.randn(C.shape[0])
    # take pseudo-inverse, remove projection within C
    C_pinv = torch.linalg.pinv(C)
    ortho_input_vec = gen_input_vec - C @ C_pinv @ gen_input_vec
    # test projection
    assert torch.allclose(C_pinv @ ortho_input_vec, torch.zeros(1), atol=1e-7)
    # normalize
    ortho_input_vec = ortho_input_vec / torch.linalg.norm(ortho_input_vec)
    print("Sampled Off-Manifold Input Vector (orthogonal to C)")

    def simulate_lorenz(n_trials, perturbed_ratio=0.0):
        y_list, vel_list, z_list, u_list = [], [], [], []

        burn_in_time = 1.0
        dt_integration = 0.005
        steps_per_bin = int(dt / dt_integration)

        kick_prob = 0.05
        kick_magnitude = 5.0
        off_manifold_kick_magnitude = 5.0
        n_inputs = n_latents_true

        for _ in range(n_trials):
            # Check if this specific trial is allowed to have kicks
            is_perturbed_trial = np.random.rand() < perturbed_ratio

            state = np.random.randn(3) * 5.0 + np.array([0, 0, 20])
            t_burn = np.arange(0, burn_in_time, dt_integration)
            burn_states = odeint(lorenz_dynamics, state, t_burn)
            state = burn_states[-1]

            trial_z, trial_y, trial_v, trial_u = [], [], [], []

            for b in range(n_bins):
                u_bin = np.zeros(n_inputs)
                off_manifold_inputs = torch.zeros(n_neurons)

                # Apply Kick
                if is_perturbed_trial and (np.random.rand() < kick_prob):
                    u_bin = direction_vec * kick_magnitude
                    state = state + (B_in.numpy() @ u_bin)
                    off_manifold_inputs = ortho_input_vec * off_manifold_kick_magnitude

                # Integrating bin with odeint
                t_bin = np.linspace(0, dt, steps_per_bin + 1)
                bin_states = odeint(lorenz_dynamics, state, t_bin)

                state = bin_states[-1]  # End state becomes start state for next bin

                # Store Bin Stats (average over the fine steps)
                z_bin_raw = np.mean(bin_states[:-1], axis=0)
                z_bin_scaled = (z_bin_raw - np.array([0, 0, 25.0])) / 10.0
                z_bin_shifted = z_bin_scaled + np.array([2, 2, 2])
                z_tensor = torch.tensor(z_bin_shifted, dtype=torch.float32)

                # Activation
                pre_rate_activation = (
                    torch.matmul(z_tensor, C.T) + d + off_manifold_inputs
                )
                # clipping to prevent explosion
                rate = torch.clamp(torch.exp(pre_rate_activation), max=1000.0) * dt
                spikes = torch.poisson(rate)

                vel = torch.matmul(z_tensor, C_vel.T)

                trial_z.append(z_tensor)
                trial_y.append(spikes)
                trial_v.append(vel)
                trial_u.append(torch.tensor(u_bin, dtype=torch.float32))

            y_list.append(torch.stack(trial_y))
            vel_list.append(torch.stack(trial_v))
            z_list.append(torch.stack(trial_z))
            u_list.append(torch.stack(trial_u))

        return (
            torch.stack(y_list),
            torch.stack(vel_list),
            torch.stack(z_list),
            torch.stack(u_list),
        )

    # Generate Split Ratios
    print("Simulating Training Set (30% Perturbed)...")
    train_y, train_v, train_z, train_u = simulate_lorenz(
        n_img_train, perturbed_ratio=0.3
    )

    print("Simulating Validation Set (30% Perturbed)...")
    valid_y, valid_v, valid_z, valid_u = simulate_lorenz(
        n_img_valid, perturbed_ratio=0.3
    )

    print("Simulating Test: Baseline (0% Perturbed)...")
    test_base_y, test_base_v, test_base_z, test_base_u = simulate_lorenz(
        n_img_test, perturbed_ratio=0.0
    )

    print("Simulating Test: Adaptation (100% PERTURBED)...")
    test_adapt_y, test_adapt_v, test_adapt_z, test_adapt_u = simulate_lorenz(
        n_img_test, perturbed_ratio=1.0
    )

    print("Simulating Test: Washout (0% Perturbed)...")
    test_wash_y, test_wash_v, test_wash_z, test_wash_u = simulate_lorenz(
        n_img_test, perturbed_ratio=0.0
    )

    # print some train set stats
    print("Train Set Stats:")
    print(
        f"Min/Max/Mean Firing Rate: \
            {train_y.min().item()}/{train_y.max().item()}\
                /{train_y.float().mean().item():.2f}"
    )
    print(
        f"Min/Max/Mean Velocity: \
            {train_v.min().item()}/{train_v.max().item()}\
                /{train_v.float().mean().item():.2f}"
    )
    print(
        f"Min/Max/Mean Latent: \
            {train_z.min().item()}/{train_z.max().item()}\
                /{train_z.float().mean().item():.2f}"
    )

    # Save format
    common_data = {
        "params": {
            "C": C,
            "d": d,
            "C_vel": C_vel,
            "B_in": B_in,
            "direction_vec": torch.tensor(
                direction_vec, dtype=torch.float32
            ),  # Saving ground truth direction
            "ortho_input_vec": torch.tensor(
                ortho_input_vec, dtype=torch.float32
            ),  # Saving off-manifold input vector
        }
    }

    def save_split(y, v, z, u, name):
        # Create a boolean mask of shape (n_trials, n_bins)
        kick_mask = u.abs().sum(dim=-1) > 0

        # Create a list of lists with exact integer indices of the kicks per trial
        kick_bins = [torch.where(mask)[0].tolist() for mask in kick_mask]

        torch.save(
            {
                "y_obs": y,
                "velocity": v,
                "true_latents": z,
                "true_inputs": u,
                "kick_mask": kick_mask,
                "kick_bins": kick_bins,
                "n_neurons_obs": n_neurons,
                "n_time_bins_enc": n_bins,
                "is_perturbed": (
                    u.abs().sum(dim=(1, 2)) > 0
                ),  # Boolean tensor (n_trials,)
                **common_data,
            },
            os.path.join(out_dir, f"{name}_{bin_sz_ms}ms.pt"),
        )

    save_split(train_y, train_v, train_z, train_u, "data_train")
    save_split(valid_y, valid_v, valid_z, valid_u, "data_valid")

    save_split(test_base_y, test_base_v, test_base_z, test_base_u, "data_test_baseline")
    save_split(
        test_adapt_y, test_adapt_v, test_adapt_z, test_adapt_u, "data_test_adaptation"
    )
    save_split(test_wash_y, test_wash_v, test_wash_z, test_wash_u, "data_test_washout")

    # Combined 'data_test' defaults to Baseline
    save_split(test_base_y, test_base_v, test_base_z, test_base_u, "data_test")

    print(f"Saved {dataset_name} to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bin_sz", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    set_seed(args.seed)
    generate_lorenz_dataset(os.path.dirname(os.path.abspath(__file__)), args.bin_sz)
