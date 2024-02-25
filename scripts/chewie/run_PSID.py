import numpy as np
import h5py
from tqdm import tqdm
import PSID # v.1.1.0, before IPSID

n_factors = 40
n_beh_factors = 20

experiments = [
    "Chewie_CO_FF_2016-09-15",
    "Chewie_CO_FF_2016-09-21",
    "Chewie_CO_FF_2016-10-05",
    "Chewie_CO_FF_2016-10-07",
    "Mihili_CO_FF_2014-02-03",  # *
    "Mihili_CO_FF_2014-02-17",  # + (only BL)
    "Mihili_CO_FF_2014-02-18",
    "Mihili_CO_FF_2014-03-07",
]

def r2_score(y_true, y_pred):
    y_true = y_true.reshape(-1, y_true.shape[-1])
    y_pred = y_pred.reshape(-1, y_pred.shape[-1])
    SS_res =  np.sum((y_true - y_pred)**2, axis=0)
    SS_tot = np.sum((y_true - np.mean(y_true, axis=0))**2, axis=0)
    return np.mean(1 - SS_res/SS_tot)

def save_results(f,area,train_outputs, test_outputs):
    if f'train_{area}_psid_pred' in f:
        del f[f'train_{area}_psid_pred']
    if f'test_{area}_psid_pred' in f:
        del f[f'test_{area}_psid_pred']
    f.create_dataset(f'train_{area}_psid_pred', data=train_outputs)
    f.create_dataset(f'test_{area}_psid_pred', data=test_outputs)

summary_dict = {}   
for short_dataset_name in tqdm(experiments):

    summary_dict[short_dataset_name] = {}

    for area in ['all','PMd','M1']:

        dataset_name = f'{short_dataset_name}_session_vel_{area}_spikes_go'
        loadpath = f'/disk/scratch2/nkudryas/BAND-torch/datasets/{dataset_name}.h5'

        data = {}
        with h5py.File(loadpath, 'r') as h5file:

            train_data=h5file['train_recon_data'][()].astype(np.int32)
            valid_data=h5file['valid_recon_data'][()].astype(np.int32)
            train_behavior=h5file['train_behavior'][()].astype(np.float32)
            valid_behavior=h5file['valid_behavior'][()].astype(np.float32)
            train_epoch=h5file['train_epoch'][()].astype(np.int32)
            valid_epoch=h5file['valid_epoch'][()].astype(np.int32)
            train_inds=h5file['train_inds'][()].astype(np.int32)
            valid_inds=h5file['valid_inds'][()].astype(np.int32)

            train_target_direction=h5file['train_target_direction'][()].astype(np.float32)
            valid_target_direction=h5file['valid_target_direction'][()].astype(np.float32)

            for key in h5file.keys():
                data[key] = h5file[key][()]
            
        # PSID(neural, behaviour, latents dimensions,
        # latent states to extract in the first stage,
        # future and past horizon)

        # run here with 40 dim, where there are some oscillations
        # mask = np.ones_like(data['train_epoch'],dtype=bool) # train on all epochs
        # mask = train_epoch == 1 # train on the AD epoch only
        # model = PSID.PSID(
        #     [d for d in train_data[mask]], [d for d in train_behavior[mask]], 
        #     n_factors, n_beh_factors, 10)
        
        # mask = data['train_epoch'][:] == 1 # train on the AD epoch only
        model = PSID.PSID(
            [d for d in train_data], [d for d in train_behavior], 
            n_factors, n_beh_factors, 10
        )

        train_outputs =  np.array([model.predict(train_data[trial])[0] for trial in range(train_data.shape[0])])
        test_outputs = np.array([model.predict(valid_data[trial])[0] for trial in range(valid_data.shape[0])])

        final_r2 = r2_score(valid_behavior, test_outputs).item()
        summary_dict[short_dataset_name][f"{area}_R2_all"] = np.round(100*final_r2,1) 

        for e, epoch_name in enumerate(['BL','AD','WO']):
            final_r2 = r2_score(valid_behavior[valid_epoch==e], test_outputs[valid_epoch==e]).item()
            summary_dict[short_dataset_name][f"{area}_R2_{epoch_name}"] = np.round(100*final_r2,1)       

        # save predictions
        results_path = f'./results/{short_dataset_name}.h5'
        with h5py.File(results_path, 'a') as f:
            save_results(f,area,train_outputs, test_outputs)

# save summary
with open("./results/PSID_decoder_R2.csv", "w") as f:
    get_column_names = summary_dict[experiments[0]].keys()
    f.write("Dataset,")
    for key in get_column_names:
        f.write(f"{key},\t")
    f.write('\n')
    for key in summary_dict.keys():
        f.write(f"{key},\t")
        for key2 in get_column_names:
            f.write(f"{summary_dict[key][key2]},\t")
        f.write('\n')