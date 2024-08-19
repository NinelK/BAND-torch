import numpy as np
import h5py
from tqdm import tqdm

from lfads_torch.metrics import r2_score

BIN_SIZE = 0.01 # seconds, i.e. 10ms

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

def save_results(f,train_outputs, test_outputs):
    if f'train_all_avg_per_epoch_pred' in f:
        del f[f'train_all_avg_per_epoch_pred']
    if f'test_all_avg_per_epoch_pred' in f:
        del f[f'test_all_avg_per_epoch_pred']
    f.create_dataset(f'train_all_avg_per_epoch_pred', data=train_outputs)
    f.create_dataset(f'test_all_avg_per_epoch_pred', data=test_outputs)

summary_dict = {}   
for short_dataset_name in tqdm(experiments):

    print(short_dataset_name)

    summary_dict[short_dataset_name] = {}

    # area = 'PMd'
    # dataset_name = f'{short_dataset_name}_session_vel_{area}_spikes_go'
    # loadpath = f'/disk/scratch2/nkudryas/BAND-torch/datasets/{dataset_name}.h5'

    loadpath = f'/disk/scratch2/nkudryas/BAND-torch/results/{short_dataset_name}.h5'

    data = h5py.File(loadpath, 'r')

    dset='train'
    vel = data[f'{dset}_behavior'][:]
    target_direction = data[f'{dset}_target_direction'][:]
    epoch = data[f'{dset}_epoch'][:]
    testQ = np.zeros_like(epoch)

    dset='valid'
    vel = np.concatenate([vel,data[f'{dset}_behavior'][:]])
    target_direction = np.concatenate([target_direction,data[f'{dset}_target_direction'][:]])
    epoch = np.concatenate([epoch,data[f'{dset}_epoch'][:]])
    testQ = np.concatenate([testQ,np.ones_like(data[f'{dset}_epoch'][:])])

    pos = np.cumsum(vel*BIN_SIZE,1)

    data.close()
    #####

    dir_index = np.array([
            sorted(set(target_direction)).index(i) for i in target_direction
        ])
    
    avg_vel = np.zeros_like(vel)
    avg_vel_per_epoch = np.zeros_like(vel)
    trial_coverage = np.zeros(vel.shape[0],dtype=bool)
    for d in range(8):
        mask = d==dir_index
        avg_vel[mask] = vel[mask].mean(0)
        for e in [0,1,2]:
            mask = (d==dir_index) & (e==epoch)
            avg_vel_per_epoch[mask] = vel[mask & (testQ==0)].mean(0)
            trial_coverage[mask] = 1
    assert trial_coverage.all()
    
    mask_test = testQ==1 # evaluate on test
    R2_vel = r2_score(avg_vel_per_epoch[mask_test], vel[mask_test])
    summary_dict[short_dataset_name][f'R2_all'] = np.round(R2_vel*100,1) 

    for e, epoch_name in enumerate(['BL','AD','WO']):
        mask = (e==epoch) & mask_test
        R2_vel = r2_score(avg_vel_per_epoch[mask], vel[mask])
        summary_dict[short_dataset_name][f'R2_{epoch_name}'] = np.round(R2_vel*100,1) 

    # save predictions
    results_path = f'./results/{short_dataset_name}.h5'
    with h5py.File(results_path, 'a') as f:
        save_results(f,avg_vel_per_epoch[testQ==0], avg_vel_per_epoch[testQ==1])

# save summary
with open(f"./results/avg_per_epoch_R2.csv", "w") as f:
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