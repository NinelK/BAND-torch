## Test if copies of the data in source and results are matching

import numpy as np
import h5py

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

keys = ['train_behavior','valid_behavior','train_target_direction','valid_target_direction',
        'train_epoch','valid_epoch','train_inds','valid_inds']


for short_dataset_name in experiments:
    
    loadpath_results = f'/disk/scratch2/nkudryas/BAND-torch/results/{short_dataset_name}.h5'
    results_data = h5py.File(loadpath_results, 'r')

    for area in ['all','M1','PMd']:
        dataset_name = f'{short_dataset_name}_session_vel_{area}_spikes_go'
        loadpath = f'/disk/scratch2/nkudryas/BAND-torch/datasets/{dataset_name}.h5'

        source_data = h5py.File(loadpath, 'r')

        for key in keys:

            assert np.allclose(results_data[key][:], source_data[key][:]), f'{key} is not consistent for {short_dataset_name}'


