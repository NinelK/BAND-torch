import numpy as np
import pyaldata

raw_data_dir = "/disk/data1/motor_cortex/perich_2018_binned/"

experiments = [
    "Chewie_CO_FF_2016-09-15.mat",
    "Chewie_CO_FF_2016-09-21.mat",
    "Chewie_CO_FF_2016-10-05.mat",
    "Chewie_CO_FF_2016-10-07.mat",
    "Mihili_CO_FF_2014-02-03.mat",  # *
    "Mihili_CO_FF_2014-02-17.mat",  # + (only BL)
    "Mihili_CO_FF_2014-02-18.mat",
    "Mihili_CO_FF_2014-03-07.mat",
]

trial_len = 100


# load all velocities for all sessions
Vel, Epoch, Target, Trial = {}, {}, {}, {}
idx_go_cue, idx_movement_on, idx_peak_speed, idx_trial_end = {}, {}, {}, {}
for spike_data_dir in experiments:
    pd_data = pyaldata.io.mat2dataframe(
        path=raw_data_dir + spike_data_dir, shift_idx_fields=True
    )

    # trials must be successful to be included
    selected_trials = pd_data.loc[
        (pd_data.result == "R")
        & (
            pd_data["vel"].apply(lambda x: np.all(np.abs(x) < 100))
            & (pd_data["idx_movement_on"] + trial_len < pd_data["idx_trial_end"])
        )
    ]

    Vel[spike_data_dir] = np.asarray(
        [
            d[f"vel"][int(d.idx_movement_on) : int(d.idx_movement_on) + trial_len, :]
            for _, d in selected_trials.iterrows()
        ]
    )

    Target[spike_data_dir] = np.asarray(
        [d["target_direction"] for _, d in selected_trials.iterrows()]
    )

    Trial[spike_data_dir] = np.asarray(
        [d["trial_id"] for _, d in selected_trials.iterrows()]
    )

    # epoch (BL/AD/WO) for each trial
    Epoch[spike_data_dir] = np.asarray(
        [d["epoch"] for _, d in selected_trials.iterrows()]
    )

    idx_go_cue[spike_data_dir] = np.asarray(
        [d["idx_go_cue"] for _, d in selected_trials.iterrows()]
    )

    idx_movement_on[spike_data_dir] = np.asarray(
        [d["idx_movement_on"] for _, d in selected_trials.iterrows()]
    )

    idx_peak_speed[spike_data_dir] = np.asarray(
        [d["idx_peak_speed"] for _, d in selected_trials.iterrows()]
    )

    idx_trial_end[spike_data_dir] = np.asarray(
        [d["idx_trial_end"] for _, d in selected_trials.iterrows()]
    )

# dump Vel, Epoch, Target, Trial in file
np.savez(
    "behavioral_data/vel_epoch_target_trial.npz",
    Vel=Vel,
    Epoch=Epoch,
    Target=Target,
    Trial=Trial,
)
# dump idx_go_cue, idx_movement_on, idx_peak_speed, idx_trial_end
np.savez(
    "behavioral_data/idx_fields.npz",
    idx_go_cue=idx_go_cue,
    idx_movement_on=idx_movement_on,
    idx_peak_speed=idx_peak_speed,
    idx_trial_end=idx_trial_end,
)


# load all velocities for all sessions
Angle, Epoch_short, Target_short, Trial_short, Sign = {}, {}, {}, {}, {}

for spike_data_dir in experiments:
    pd_data = pyaldata.io.mat2dataframe(
        path=raw_data_dir + spike_data_dir, shift_idx_fields=True
    )
    selected_trials = pd_data.loc[
        (pd_data["idx_movement_on"] < pd_data["idx_trial_end"])
    ]

    perturb_info = np.array([d for d in selected_trials["perturbation_info"][:]])
    assert np.all(
        perturb_info == perturb_info[0]
    ), "All trials must have the same perturbation info for this dataset."
    Sign[spike_data_dir] = np.sign(perturb_info[0, 1])

    pos_mov_on = np.asarray(
        [d[f"pos"][int(d.idx_movement_on) - 1] for _, d in selected_trials.iterrows()]
    )

    pos_end = np.zeros_like(pos_mov_on)

    mask = np.asarray(
        [d.idx_peak_speed < d.idx_trial_end for _, d in selected_trials.iterrows()]
    )

    print(mask.shape, pos_end.shape)

    pos_end[mask] = np.asarray(
        [d[f"pos"][int(d.idx_peak_speed) - 1] for _, d in selected_trials.iterrows()]
    )[mask]

    pos_end[~mask] = np.asarray(
        [d[f"pos"][int(d.idx_trial_end) - 1] for _, d in selected_trials.iterrows()]
    )[~mask]

    Angle[spike_data_dir] = np.arctan2(
        pos_end[:, 1] - pos_mov_on[:, 1], pos_end[:, 0] - pos_mov_on[:, 0]
    )

    Target_short[spike_data_dir] = np.asarray(
        [d["target_direction"] for _, d in selected_trials.iterrows()]
    )

    Trial_short[spike_data_dir] = np.asarray(
        [d["trial_id"] for _, d in selected_trials.iterrows()]
    )

    # epoch (BL/AD/WO) for each trial
    Epoch_short[spike_data_dir] = np.asarray(
        [d["epoch"] for _, d in selected_trials.iterrows()]
    )

# dump Angle, Epoch_short, Target_short, Trial_short, Sign in file
np.savez(
    "behavioral_data/angle_epoch_target_trial_sign.npz",
    Angle=Angle,
    Epoch_short=Epoch_short,
    Target_short=Target_short,
    Trial_short=Trial_short,
    Sign=Sign,
)
