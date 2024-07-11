import numpy as np
import pyaldata
import h5py
import os

raw_data_dir = (
    "/disk/scratch2/nkudryas/motor_cortex/perich_2018_binned/"
)
data_save_dir = "./datasets/"

experiments = [
    "Chewie_CO_FF_2016-09-15.mat",
    "Chewie_CO_FF_2016-09-21.mat",
    "Chewie_CO_FF_2016-10-05.mat",
    "Chewie_CO_FF_2016-10-07.mat",  # the best!
    # "Chewie_CO_VR_2016-09-09.mat",
    # "Chewie_CO_VR_2016-09-12.mat",
    # "Chewie_CO_VR_2016-09-14.mat",
    # "Chewie_CO_VR_2016-10-06.mat",
    "Mihili_CO_FF_2014-02-03.mat",  # *
    "Mihili_CO_FF_2014-02-17.mat",  # + (only BL)
    "Mihili_CO_FF_2014-02-18.mat",
    "Mihili_CO_FF_2014-03-07.mat",
    # "Mihili_CO_VR_2014-03-03.mat",
    # "Mihili_CO_VR_2014-03-04.mat",
    # "Mihili_CO_VR_2014-03-06.mat",
]

trial_dur = None # will be set to the shortest trial duration
n_folds = 5
alignment = "go"
behaviour_key="vel"
full_behaviour_key="pos"
dataset_type = "session" # "BL", "AD", "WO", "session"
data_type = 'spikes' # spikes or MUA

summary_dict = {}

def etype(epoch):
    if epoch == "BL":
        return 0
    elif epoch == "AD":
        return 1
    elif epoch == "WO":
        return 2
    else:
        return -1

for experiment in experiments:
    spike_data_dir = experiment
    pd_data = pyaldata.io.mat2dataframe(
        path=raw_data_dir +spike_data_dir,
        shift_idx_fields=True,
    )
    print(spike_data_dir)
    for activity_key in [f"all_{data_type}",f"M1_{data_type}",f"PMd_{data_type}"]:
        
        # create pseudo-MUA data
        if 'MUA' in activity_key:
            areas = ['M1', 'PMd']
            n = {area: len(np.unique(pd_data[f"{area}_unit_guide"][0][:,0])) for area in areas}
            MUAs = {area: [] for area in areas}
            for area in areas:
                for i in range(len(pd_data[f"{area}_spikes"])):
                    this_trial_MUA = np.zeros((pd_data[f"{area}_spikes"][i].shape[0], n[area]))
                    n_mua = -1
                    ug = pd_data[f'{area}_unit_guide'][i]
                    unique_units = np.unique(ug[:,0])
                    min_el_id = 1
                    assert ug[0,1] == min_el_id, "first neuron is not the first on the first electrode? wrong unit guide"
                    for j in range(pd_data[f"{area}_spikes"][i].shape[1]):
                        if ug[j,1] == min_el_id: # if the first neuron -- move on
                            n_mua += 1
                            this_trial_MUA[:,n_mua] = pd_data[f"{area}_spikes"][i][:,j]
                            if n_mua + 1 >= n[area]:
                                break
                            min_el_id = ug[:,1][ug[:,0] == unique_units[n_mua+1]].min()
                            if (min_el_id!=1) & (i==0): # report only for the first trial
                                print(f'electrode {unique_units[n_mua+1]} has no neuron 1, only {min_el_id}')
                        else: # for second and further neurons on the same electrode -- add spikes
                            this_trial_MUA[:,n_mua] += pd_data[f"{area}_spikes"][i][:,j]
                    MUAs[area].append(this_trial_MUA)
                    assert (n_mua + 1) == n[area], f"n_mua = {n_mua}, n[{area}] = {n[area]}, not all neurons used?"
            pd_data["M1_MUA"] = MUAs['M1']
            pd_data["PMd_MUA"] = MUAs['PMd']
        
        if activity_key == f"all_{data_type}":
            pd_data[activity_key] = [
                np.concatenate([m1, pmd], axis=1)
                for m1, pmd in zip(pd_data[f"M1_{data_type}"], pd_data[f"PMd_{data_type}"])
            ]
    
        if dataset_type == "session":  # all epochs
            successful_trials = pd_data.loc[(pd_data.result == "R")]
        else:
            successful_trials = pd_data.loc[
                (pd_data.result == "R") & (pd_data.epoch == dataset_type)
            ]
        
        longest_trial = None
        if alignment == "mov":
            shortest_trial = int(
                (
                    successful_trials.idx_trial_end
                    - successful_trials.idx_movement_on
                ).min()
            )
            start_key = "idx_movement_on"
        elif alignment == "go":
            shortest_trial = int(
                (successful_trials.idx_trial_end - successful_trials.idx_go_cue).min()
            )
            longest_trial = int(
                (successful_trials.idx_trial_end - successful_trials.idx_go_cue).max()
            )
            start_key = "idx_go_cue"
        elif alignment == "target":
            shortest_trial = int(
                (
                    successful_trials.idx_trial_end - successful_trials.idx_target_on
                ).min()
            )
            start_key = "idx_target_on"
        print(
            f"Shortest : {shortest_trial}, longest: {longest_trial}, succesful trials: {len(successful_trials)}/{len(pd_data)}"
        )
        if activity_key == f"all_{data_type}":
            trial_dur = shortest_trial
        else:
            shortest_trial = trial_dur  # overwrite

        # exclude bad trials
        behaviours = np.asarray(
            [
                d[behaviour_key][d[start_key] : d[start_key] + shortest_trial, :]
                for i, d in successful_trials.iterrows()
            ]
        )
        trials_to_exclude = np.unique(np.where(np.abs(behaviours) > 100)[0])
        trial_mask = np.arange(behaviours.shape[0])
        trial_mask = np.delete(trial_mask,trials_to_exclude)
        print('Exclude trials with bad velocity: ',trials_to_exclude)
        # EXCLUDE
        successful_trials = successful_trials.iloc[trial_mask]
        print('Number of trials after excluding: ',len(successful_trials))

        order = np.arange(len(successful_trials)) % n_folds

        for fold in range(n_folds):            
            train_df = successful_trials.iloc[order != fold]
            valid_df = successful_trials.iloc[order == fold]

            train_df.idx_movement_on = train_df.idx_movement_on.astype(int)
            valid_df.idx_movement_on = valid_df.idx_movement_on.astype(int)

            train_data = np.asarray(
                [
                    d[activity_key][d[start_key] : d[start_key] + shortest_trial, :]
                    for i, d in train_df.iterrows()
                ]
            )
            valid_data = np.asarray(
                [
                    d[activity_key][d[start_key] : d[start_key] + shortest_trial, :]
                    for i, d in valid_df.iterrows()
                ]
            )

            print(train_data.shape, valid_data.shape)

            key = experiment + "_cv" + str(fold)
            if activity_key == f"all_{data_type}":
                summary_dict[key] = {
                    "BL_trials": len(successful_trials.loc[successful_trials.epoch == "BL"]),
                    "AD_trials": len(successful_trials.loc[successful_trials.epoch == "AD"]),
                    "WO_trials": len(successful_trials.loc[successful_trials.epoch == "WO"]),
                    "train_trials": len(train_df),
                    "valid_trials": len(valid_df),
                    "duration": shortest_trial,
                    "all_neurons": train_data.shape[2]
                    }
            else:
                summary_dict[key][f"{activity_key.split('_')[0]}_neurons"] = train_data.shape[2]
    
            train_target_direction = np.asarray(
                [d["target_direction"] for i, d in train_df.iterrows()]
            )
            valid_target_direction = np.asarray(
                [d["target_direction"] for i, d in valid_df.iterrows()]
            )

            train_trial = np.asarray([d["trial_id"] for i, d in train_df.iterrows()])
            valid_trial = np.asarray([d["trial_id"] for i, d in valid_df.iterrows()])
            
            train_behaviours = np.asarray(
                [
                    d[behaviour_key][d[start_key] : d[start_key] + shortest_trial, :]
                    for i, d in train_df.iterrows()
                ]
            )
            valid_behaviours = np.asarray(
                [
                    d[behaviour_key][d[start_key] : d[start_key] + shortest_trial, :]
                    for i, d in valid_df.iterrows()
                ]
            )

            train_pos = np.asarray(
                [
                    d[full_behaviour_key][
                        d[start_key] : d[start_key] + shortest_trial, :
                    ]
                    for i, d in train_df.iterrows()
                ]
            )
            valid_pos = np.asarray(
                [
                    d[full_behaviour_key][
                        d[start_key] : d[start_key] + shortest_trial, :
                    ]
                    for i, d in valid_df.iterrows()
                ]
            )

            origin = train_pos[:, 0].mean(0)
            train_pos -= origin
            valid_pos -= origin

            train_epoch = np.asarray(
                [etype(d["epoch"]) for i, d in train_df.iterrows()]
            )
            valid_epoch = np.asarray(
                [etype(d["epoch"]) for i, d in valid_df.iterrows()]
            )

            data_dir = (
                data_save_dir
                + spike_data_dir[:-4]
                + "_"
                + dataset_type
                + "_"
                + behaviour_key
                + "_"
                + activity_key
                + "_"
                + alignment
                + "_cv"
                + str(fold)
            )

            filename = data_dir + f".h5"

            with h5py.File(filename, 'w') as h5file:
                # variables needed for training
                h5file.create_dataset('train_encod_data', data=train_data)
                h5file.create_dataset('valid_encod_data', data=valid_data)
                h5file.create_dataset('train_recon_data', data=train_data)
                h5file.create_dataset('valid_recon_data', data=valid_data)
                h5file.create_dataset('train_behavior', data=train_behaviours)
                h5file.create_dataset('valid_behavior', data=valid_behaviours)
                # variables needed for post analysis
                h5file.create_dataset('train_inds', data=train_trial)
                h5file.create_dataset('valid_inds', data=valid_trial)
                h5file.create_dataset('train_epoch', data=train_epoch)
                h5file.create_dataset('valid_epoch', data=valid_epoch)
                h5file.create_dataset('train_pos', data=train_pos)
                h5file.create_dataset('valid_pos', data=valid_pos)
                h5file.create_dataset('train_vel', data=train_behaviours)
                h5file.create_dataset('valid_vel', data=valid_behaviours)
                h5file.create_dataset('train_target_direction', data=train_target_direction)
                h5file.create_dataset('valid_target_direction', data=valid_target_direction)

            short_dataset_name = spike_data_dir[:-4]

            # check if results file is there
            results_path = f'./results/{short_dataset_name}_cv{fold}.h5'
            if not os.path.exists(results_path):
                with h5py.File(results_path, 'w') as f:               
                    f.create_dataset('train_behavior', data=train_behaviours)
                    f.create_dataset('valid_behavior', data=valid_behaviours)
                    f.create_dataset('train_target_direction', data=train_target_direction)
                    f.create_dataset('valid_target_direction', data=valid_target_direction)
                    f.create_dataset('train_epoch', data=train_epoch)
                    f.create_dataset('valid_epoch', data=valid_epoch)
                    f.create_dataset('train_inds', data=train_trial)
                    f.create_dataset('valid_inds', data=valid_trial)

            

# save summary
with open("./results/dataset_summary.csv", "w") as f:
    get_column_names = summary_dict[f"{experiments[0]}_cv0"].keys()
    f.write("Dataset,")
    for key in get_column_names:
        f.write(f"{key},\t")
    f.write('\n')
    for key in summary_dict.keys():
        f.write(f"{key},\t")
        for key2 in get_column_names:
            f.write(f"{summary_dict[key][key2]},\t")
        f.write('\n')