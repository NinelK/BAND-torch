import numpy as np
import h5py
from tqdm import tqdm
from cebra import CEBRA, KNNDecoder
import sklearn.metrics

max_iterations = 10000
batch_size = 512
output_dimension = 32
lr = 0.0001
T = 1

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
    if f'train_{area}_{model_name}_pred' in f:
        del f[f'train_{area}_{model_name}_pred']
    if f'test_{area}_{model_name}_pred' in f:
        del f[f'test_{area}_{model_name}_pred']
    f.create_dataset(f'train_{area}_{model_name}_pred', data=train_outputs)
    f.create_dataset(f'test_{area}_{model_name}_pred', data=test_outputs)

lfads_results_paths = {{} for short_dataset_name in experiments}
band_results_paths = {{} for short_dataset_name in experiments}

lfads_results_paths['Chewie_CO_FF_2016-10-05']['all'] = '../runs/band-paper/chewie_10_05_lfads/lfads_100f_kl1_studentT_bs256'
band_results_paths['Chewie_CO_FF_2016-10-05']['all'] = '../runs/band-paper/chewie_10_05_band/band_100f_kl1_studentT_bs256'

summary_dict_lfads, summary_dict_band = {}, {}   
for short_dataset_name in tqdm(experiments):

    summary_dict_lfads[short_dataset_name] = {}
    summary_dict_band[short_dataset_name] = {}

    for area in ['all','PMd','M1']:

        dataset_name = f'{short_dataset_name}_session_vel_{area}_spikes_go'
        loadpath = f'/disk/scratch2/nkudryas/BAND-torch/datasets/{dataset_name}.h5'

        with h5py.File(loadpath, 'r') as h5file:

            train_data=h5file['train_recon_data'][()].astype(np.float32)
            valid_data=h5file['valid_recon_data'][()].astype(np.float32)
            train_behavior=h5file['train_behavior'][()].astype(np.float32)
            valid_behavior=h5file['valid_behavior'][()].astype(np.float32)
            train_epoch=h5file['train_epoch'][()].astype(np.float32)
            valid_epoch=h5file['valid_epoch'][()].astype(np.float32)
            train_inds=h5file['train_inds'][()].astype(np.float32)
            valid_inds=h5file['valid_inds'][()].astype(np.float32)

            train_target_direction=h5file['train_target_direction'][()].astype(np.float32)
            valid_target_direction=h5file['valid_target_direction'][()].astype(np.float32)
            
        
        # load LFADS model
            
        overrides={
            "datamodule": dataset_name,
            "model": dataset_name.replace('_M1', '').replace('_PMd',''),
            "model.encod_data_dim": sys.argv[3],
            "model.behavior_weight": sys.argv[4],
        }
        config_path="../configs/single.yaml"

        # Compose the train config with properly formatted overrides
        config_path = Path(config_path)
        overrides = [f"{k}={v}" for k, v in flatten(overrides).items()]
        with hydra.initialize(
            config_path=config_path.parent,
            job_name="get_weights",
            version_base="1.1",
        ):
            config = hydra.compose(config_name=config_path.name, overrides=overrides)

        # Instantiate `LightningDataModule` and `LightningModule`
        datamodule = instantiate(config.datamodule, _convert_="all")
        model = instantiate(config.model)

        ckpt_path = f'{model_dest}/lightning_checkpoints/last.ckpt'
        model.load_state_dict(torch.load(ckpt_path)["state_dict"])
            
        # run ridge regression
            

        final_r2 = r2_score(valid_behavior, test_outputs).item()
        summary_dict[short_dataset_name][f"{area}_R2_all"] = np.round(100*final_r2,1) 

        for e, epoch_name in enumerate(['BL','AD','WO']):
            final_r2 = r2_score(valid_behavior[valid_epoch==e], test_outputs[valid_epoch==e]).item()
            summary_dict[short_dataset_name][f"{area}_R2_{epoch_name}"] = np.round(100*final_r2,1)       

        # save predictions
        results_path = f'./results/{short_dataset_name}.h5'
        with h5py.File(results_path, 'a') as f:
            save_results(f,area,train_outputs, test_outputs)

# # save summary
# with open("./results/lfads_decoder_R2.csv", "w") as f:
#     get_column_names = summary_dict[experiments[0]].keys()
#     f.write("Dataset,")
#     for key in get_column_names:
#         f.write(f"{key},\t")
#     f.write('\n')
#     for key in summary_dict.keys():
#         f.write(f"{key},\t")
#         for key2 in get_column_names:
#             f.write(f"{summary_dict[key][key2]},\t")
#         f.write('\n')