import argparse
import os
import h5py

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


def generate_lorenz_dataset(base_dir, bin_sz_ms=10, delay_bins=0, n_behavior=2, behavior_noise_std=0.1):
    """
    Synthetic Lorenz attractor dataset for BAND training & evaluation.

    The simulation works as follows:
    1. A 3D Lorenz system evolves over time to produce true latent states (z).
    2. Latent states are mapped to neural firing rates via: rate = exp(C @ z + d),
       then Poisson spiking produces observed spike counts (y_obs / encod_data).
    3. Behavioral output ('velocity') is computed as: vel = C_vel @ z + noise.
    4. On "perturbed" trials, random kicks are injected:
       - On-manifold kicks shift the Lorenz state via B_in @ (direction_vec * magnitude).
       - Off-manifold kicks add activity orthogonal to C.

    n_neurons       : Number of simulated neurons (channels in the spiking data).
    n_latents_true  : Dimensionality of the true latent Lorenz system (always 3).
    n_bins          : Number of time bins per trial.
    delay_bins > 0  : Behavior lags latents (neural activity leads movement).
    delay_bins < 0  : Behavior leads latents (movement leads neural activity).
    dt              : Duration of each time bin in seconds.
    C               : Neural readout matrix (n_neurons x n_latents_true).
                      Maps latent states to log-firing rates.
    d               : Baseline firing rate bias.
    n_behavior      : Dimensionality of the behavioral output (default 2).
    C_vel           : Velocity readout matrix (n_behavior x n_latents_true).
                      Maps latent states to behavioral velocity via a rank-deficient
                      random projection.
    behavior_noise_std : Std of additive Gaussian noise on the behavioral output.
    B_in            : Input gain matrix (n_latents_true x n_latents_true).
                      Maps the kick vector into latent state space. Set to identity
                      so kicks are applied directly to the Lorenz state.

    direction_vec   : Fixed unit vector in latent space.
                      All on-manifold kicks point in this single direction.
    ortho_input_vec : Unit vector in neural space orthogonal to C.

    Output files:
    {dataset_name}.h5       : Train + validation splits for normal training.
                            Train keys (train_*): Training trials (30% perturbed).
                            Validation trials (30% perturbed).
    {dataset_name}_test.h5  : Train + adaptation-test splits for evaluation.
                            Exact same training trials (30% perturbed).
                            Test / Adaptation trials (100% perturbed).

     _epoch: array saved in the HDF5 datasets. Boolean (1 for perturbed, 0 for unperturbed) indicating whether a perturbation input/kick occurred for that trial.
    """

    # delay_bins = -10                # number of time bins to delay the input
    
    if delay_bins>0:
        dataset_name = f"data_Synthetic_Lorenz_beh_forward_{np.abs(delay_bins)}"
    elif delay_bins==0:
        dataset_name = f"data_Synthetic_Lorenz_beh_0"
    else:
        dataset_name = f"data_Synthetic_Lorenz_beh_feedback_{np.abs(delay_bins)}"

    print(f"Generating {dataset_name} (bin_sz={bin_sz_ms}ms)...")
    out_dir = os.path.join(base_dir, "..", "..", "datasets")
    os.makedirs(out_dir, exist_ok=True)

    n_neurons = 50            
    n_latents_true = 3         
    n_bins = 100         
    max_delay_bins = 20    
    assert np.abs(delay_bins)<=max_delay_bins  
    dt = bin_sz_ms / 1000.0    

    n_trials_train= 200         # number of training trials
    n_trials_valid = 200          # number of validation trials
    n_trials_test = 200           # number of test trials

    print("Initializing Global Readout Matrices...")
    C = 1.0 * torch.randn(n_neurons, n_latents_true)      
    d = 0.0 + 0.2 * torch.randn(n_neurons)                
    C_vel = torch.randn(n_behavior, n_latents_true+1)         
    B_in = torch.eye(n_latents_true)                       

    # All on-manifold kicks will be along this single fixed direction in latent space
    print("Sampling Fixed Unidirectional Kick Vector...")
    direction_vec = torch.randn(n_latents_true).numpy()
    direction_vec = direction_vec / np.linalg.norm(direction_vec)
    print(f"Sampled Direction: {direction_vec}")

    # --- Sample another random unit vector for off-manifold input footprint ---
    # This vector lives in neural space but is orthogonal to the column space of C,
    # so it produces neural activity that the latent readout cannot explain.
    gen_input_vec = torch.randn(C.shape[0]) # whole space
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
        off_manifold_kick_magnitude = 20.0
        beh_input_weight = 0.2
        n_inputs = 1
        n_bins_sim = n_bins + 2*max_delay_bins

        for _ in range(n_trials):
            # Check if this specific trial is allowed to have kicks
            is_perturbed_trial = np.random.rand() < perturbed_ratio

            state = np.random.randn(3) * 5.0 + np.array([0, 0, 20])
            t_burn = np.arange(0, burn_in_time, dt_integration)
            burn_states = odeint(lorenz_dynamics, state, t_burn)
            state = burn_states[-1]

            trial_z, trial_y, trial_v, trial_u = [], [], [], []

            for b in range(n_bins_sim):
                u_bin = np.zeros(n_inputs)
                off_manifold_inputs = torch.zeros(n_neurons)

                # Apply Kick
                if is_perturbed_trial and (np.random.rand() < kick_prob):
                    u_bin += kick_magnitude
                    state = state + (B_in.numpy() @ (direction_vec * u_bin))
                    off_manifold_inputs = ortho_input_vec * off_manifold_kick_magnitude

                # Integrating bin with odeint
                t_bin = np.linspace(0, dt, steps_per_bin + 1)
                bin_states = odeint(lorenz_dynamics, state, t_bin)

                state = bin_states[-1]  # End state becomes start state for next bin

                # NOTE: averaging over fine steps: Within each time bin, the Lorenz ODE is integrated at a much finer resolution
                # For dt_integration = 0.005s, there are 4 fine steps per 20ms bin
                # We average those integration steps to get a single representative latent state for the bin.
                z_bin_raw = np.mean(bin_states[:-1], axis=0)
                z_bin_scaled = (z_bin_raw - np.array([0, 0, 25.0])) / 10.0
                z_bin_shifted = z_bin_scaled + np.array([2, 2, 2])
                z_tensor = torch.tensor(z_bin_shifted, dtype=torch.float32)

                pre_rate_activation = (
                    torch.matmul(z_tensor, C.T) + d + off_manifold_inputs
                )

                rate = torch.clamp(torch.exp(pre_rate_activation), max=1000.0) * dt
                spikes = torch.poisson(rate)

                vel = torch.matmul(torch.concat([z_tensor,beh_input_weight*torch.tensor(u_bin, dtype=torch.float32)]), C_vel.T) + behavior_noise_std * torch.randn(n_behavior)

                trial_z.append(z_tensor)
                trial_y.append(spikes)
                trial_v.append(vel)
                trial_u.append(torch.tensor(u_bin, dtype=torch.float32))

            y_stacked = torch.stack(trial_y)
            v_stacked = torch.stack(trial_v)
            z_stacked = torch.stack(trial_z)
            u_stacked = torch.stack(trial_u)

            
            y_sliced = y_stacked[max_delay_bins : max_delay_bins + n_bins]
            z_sliced = z_stacked[max_delay_bins : max_delay_bins + n_bins]
            u_sliced = u_stacked[max_delay_bins : max_delay_bins + n_bins]
            v_sliced = v_stacked[max_delay_bins - delay_bins : max_delay_bins - delay_bins + n_bins]

            y_list.append(y_sliced)
            vel_list.append(v_sliced)
            z_list.append(z_sliced)
            u_list.append(u_sliced)

        return (
            torch.stack(y_list),
            torch.stack(vel_list),
            torch.stack(z_list),
            torch.stack(u_list),
        )

    # Generate Split Ratios
    print("Simulating Training Set (30% Perturbed)...")
    train_y, train_v, train_z, train_u = simulate_lorenz(
        n_trials_train, perturbed_ratio=0.3
    )

    print("Simulating Validation Set (30% Perturbed)...")
    valid_y, valid_v, valid_z, valid_u = simulate_lorenz(
        n_trials_valid, perturbed_ratio=0.3
    )

    print("Simulating Test: Baseline (0% Perturbed)...")
    test_base_y, test_base_v, test_base_z, test_base_u = simulate_lorenz(
        n_trials_test, perturbed_ratio=0.0
    )

    print("Simulating Test: Adaptation (100% PERTURBED)...")
    test_adapt_y, test_adapt_v, test_adapt_z, test_adapt_u = simulate_lorenz(
        n_trials_test, perturbed_ratio=1.0
    )

    print("Simulating Test: Washout (0% Perturbed)...")
    test_wash_y, test_wash_v, test_wash_z, test_wash_u = simulate_lorenz(
        n_trials_test, perturbed_ratio=0.0
    )

    # print Dataset Stats
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
    def get_is_perturbed(u):
        return (u.abs().sum(dim=(1, 2)) > 0).numpy()

    # Convert tensors to numpy for saving in h5
    train_y_np = train_y.numpy()
    valid_y_np = valid_y.numpy()
    test_adapt_y_np = test_adapt_y.numpy()
    
    train_v_np = train_v.numpy()
    valid_v_np = valid_v.numpy()
    test_adapt_v_np = test_adapt_v.numpy()
    
    train_z_np = train_z.numpy()
    valid_z_np = valid_z.numpy()
    test_adapt_z_np = test_adapt_z.numpy()
    
    train_u_np = train_u.numpy()
    valid_u_np = valid_u.numpy()
    test_adapt_u_np = test_adapt_u.numpy()

    train_epoch = get_is_perturbed(train_u)
    valid_epoch = get_is_perturbed(valid_u)
    test_adapt_epoch = get_is_perturbed(test_adapt_u)

    def save_h5(filename, encod_data_valid, recon_data_valid, behavior_valid, epoch_valid, latents_valid, inputs_valid):
        with h5py.File(filename, 'w') as h5file:
            # Training data is consistent
            h5file.create_dataset('train_encod_data', data=train_y_np)
            h5file.create_dataset('train_recon_data', data=train_y_np)
            h5file.create_dataset('train_behavior', data=train_v_np)
            h5file.create_dataset('train_epoch', data=train_epoch)
            h5file.create_dataset('train_true_latents', data=train_z_np)
            h5file.create_dataset('train_true_inputs', data=train_u_np)
            
            # Validation/Test data may vary based on splits with different degrees of pertubation
            h5file.create_dataset('valid_encod_data', data=encod_data_valid)
            h5file.create_dataset('valid_recon_data', data=recon_data_valid)
            h5file.create_dataset('valid_behavior', data=behavior_valid)
            h5file.create_dataset('valid_epoch', data=epoch_valid)
            h5file.create_dataset('valid_true_latents', data=latents_valid)
            h5file.create_dataset('valid_true_inputs', data=inputs_valid)
            
            # Save common parameters
            h5file.create_dataset('params/C', data=C.numpy())
            h5file.create_dataset('params/d', data=d.numpy())
            h5file.create_dataset('params/C_vel', data=C_vel.numpy())
            h5file.create_dataset('params/B_in', data=B_in.numpy())
            h5file.create_dataset('params/direction_vec', data=direction_vec)
            h5file.create_dataset('params/ortho_input_vec', data=ortho_input_vec.numpy())

    # Standard Dataset
    filename = os.path.join(out_dir, f"{dataset_name}.h5")
    save_h5(filename, valid_y_np, valid_y_np, valid_v_np, valid_epoch, valid_z_np, valid_u_np)

    # Test Dataset (Adaptation)
    filename_test = os.path.join(out_dir, f"{dataset_name}_test.h5")
    save_h5(filename_test, test_adapt_y_np, test_adapt_y_np, test_adapt_v_np, test_adapt_epoch, test_adapt_z_np, test_adapt_u_np)

    print(f"Saved {dataset_name} to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bin_sz", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--delay_bins", type=int, default=0, help="Number of bins to delay behavior")
    parser.add_argument("--n_behavior", type=int, default=2, help="Behavioral output dimensionality (< n_latents_true)")
    parser.add_argument("--behavior_noise_std", type=float, default=0.1, help="Std of Gaussian noise on behavior")
    args = parser.parse_args()
    set_seed(args.seed)
    generate_lorenz_dataset(
        os.path.dirname(os.path.abspath(__file__)),
        args.bin_sz, args.delay_bins, args.n_behavior, args.behavior_noise_std
    )
