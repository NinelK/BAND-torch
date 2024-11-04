from omegaconf import OmegaConf

OmegaConf.register_new_resolver(
    "relpath", lambda p: str(Path('/disk/scratch2/nkudryas/BAND-torch/scripts/').parent / p)
) # this is required for hydra to find the config files

import numpy as np
import pandas as pd
import h5py
import torch
from hydra.utils import instantiate
from pathlib import Path
import hydra
import sys

from lfads_torch.band_utils import flatten
from lfads_torch.metrics import r2_score

import torch
from torch import nn
from lfads_torch.benchmark.biRNN_decoder import Decoder

num_epochs = 1000 # 1000 is enough
batch_size = 100

PROJECT_STR = sys.argv[1]
MODEL_STR = sys.argv[2]
DATASET_STR = sys.argv[3] #'chewie_10_07'
bin_width_sec = 0.01 # chewie
PATH = 'f"/disk/scratch2/nkudryas/BAND-torch/datasets'

best_model_dest = f"/disk/scratch2/nkudryas/BAND-torch/runs/{PROJECT_STR}/{DATASET_STR}/"
model_name = sys.argv[4]
model_dest = f"{best_model_dest}/{model_name}"

fold = None
if '_cv' in DATASET_STR:
    dataset_name, fold = DATASET_STR.split('_cv')
    print('CV fold: ',fold)
else:
    dataset_name = DATASET_STR

# Load model
encod_seq_len = sys.argv[5]
overrides={
        "datamodule": dataset_name,
        "model": MODEL_STR,
        "model.encod_seq_len": encod_seq_len,
        "model.recon_seq_len": encod_seq_len,
        "model.fac_dim": sys.argv[6],
        "model.co_dim": sys.argv[7],
        "model.encod_data_dim": sys.argv[8],
        "model.behavior_weight": sys.argv[9],
    }
if fold is not None:
    overrides["datamodule.fold"] = fold
config_path="../configs/pbt.yaml"
fac_dim = int(sys.argv[6])
co_dim = int(sys.argv[7])

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

from glob import glob

if 'pbt' in PROJECT_STR:
    # check the latest checkpoint
    checkpoint_folders = glob(model_dest+'/best_model/checkpoint*')
    ckpt_path = checkpoint_folders[-1] + '/tune.ckpt'
    model.load_state_dict(torch.load(ckpt_path)["state_dict"])
else:
    ckpt_path = f'{model_dest}/lightning_checkpoints/last.ckpt'

model.load_state_dict(torch.load(ckpt_path)["state_dict"])

# load the dataset
if fold is not None:
    datafile_pattern = config.datamodule.datafile_pattern.replace(".h5", f"_cv{fold}.h5")
else:
    datafile_pattern = config.datamodule.datafile_pattern
data_paths = sorted(glob(datafile_pattern))
# Give each session a unique file path
for sess_id, dataset_filename in enumerate(data_paths):

    session = dataset_filename.split("/")[-1].split("_")[-1].split(".")[0]
    if fold is not None:
        session = f"cv{fold}"
    
    print(dataset_filename)

    with h5py.File(dataset_filename, 'r') as f:
        train_inds, valid_inds = f['train_inds'][:], f['valid_inds'][:]
        valid_epoch = f['valid_epoch'][:]
        true_train_beh = f['train_vel'][:]
        true_valid_beh = f['valid_vel'][:]
        true_target_direction = f['valid_target_direction'][:]

    # load model components
    if 'pbt' in PROJECT_STR:
        data_path = best_model_dest + model_name + f'/best_model/lfads_output_{session}.h5'
    else:
        data_path = best_model_dest + model_name + f'/lfads_output_{session}.h5'
    with h5py.File(data_path) as f:
        rates = f["valid_output_params"][:] / bin_width_sec
        train_rates = f["train_output_params"][:] / bin_width_sec
        
    # biRNN prediction from rates
    train_data = train_rates.astype(np.float32)
    train_behavior = true_train_beh.astype(np.float32)
    valid_data = rates.astype(np.float32)
    valid_behavior = true_valid_beh.astype(np.float32)
    
    # save results in the summary file
    def save_results(f,area,mn,train_outputs, test_outputs,fac_dim=fac_dim,co_dim=co_dim,sample=''):
        if (sample=='') | (sample==0) | (sample=='0'):
            sample_str = ''
        else:
            sample_str = f'_sample{sample}'
        key = f'train_{area}_{mn}_{fac_dim}f_{co_dim}c{sample_str}_biRNN_pred'
        if key in f:
            del f[key]
        key = f'test_{area}_{mn}_{fac_dim}f_{co_dim}c{sample_str}_biRNN_pred'
        if key in f:
            del f[key]
        f.create_dataset(f'train_{area}_{mn}_{fac_dim}f_{co_dim}c{sample_str}_biRNN_pred', data=train_outputs)
        f.create_dataset(f'test_{area}_{mn}_{fac_dim}f_{co_dim}c{sample_str}_biRNN_pred', data=test_outputs)

    short_dataset_name = dataset_name.replace('_M1', '').replace('_PMd','').replace('_small','')
    if 'M1' in dataset_name:
        area = 'M1'
    elif 'PMd' in dataset_name:
        area = 'PMd'
    else:
        area = 'all'
    
    # train an RNN decoder to predict behavior from neural activity
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    N, M = train_data.shape[-1], train_behavior.shape[-1] # number of neurons, number of behavior dimensions
    T = train_data.shape[1] # number of time bins
    assert M == 2, 'only 2D behavior is expected'
    
    dropout, spike_dropout_rate = 0.2, 0.
    rnn = Decoder(input_size=N, 
                rnn_size=128,
                hidden_size=128, 
                output_size=M, 
                seq_len=T, 
                num_layers=1,
                dropout=dropout,
                spike_dropout_rate=spike_dropout_rate).to(device)

    # Loss and optimizer
    loss = nn.MSELoss()
    # higher weight decay for RNN
    optimizer = torch.optim.Adam(rnn.parameters(), lr=0.001, weight_decay=.01)

    #scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.9, verbose=False)

    # Convert numpy arrays to torch tensors
    inputs = torch.from_numpy(train_data).to(device)
    behaviors = torch.from_numpy(train_behavior).to(device)
    test_inputs = torch.from_numpy(valid_data).to(device)
    test_behaviors = torch.from_numpy(valid_behavior).to(device)

    # Train the model
    for epoch in range(num_epochs):

        batch_indices = list(range(inputs.shape[0]))
        batch = torch.from_numpy(np.random.choice(batch_indices, batch_size)).to(device)

        # print(batch)

        # Forward pass
        outputs = rnn(inputs[batch])
        cost = loss(outputs, behaviors[batch]) 

        # Backward and optimize
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()
        scheduler.step(cost)

        # print (f'Epoch [{epoch+1}/{num_epochs}], Loss: {cost.item():.4f}, Test Loss: {test_cost.item():.4f}')
        # print epoch and R2
        if (epoch+1) % 200 == 0:
            # predict on test
            rnn.eval()
            train_outputs = rnn(inputs)
            train_cost = loss(train_outputs, behaviors) 
            test_outputs = rnn(test_inputs)
            test_cost = loss(test_outputs, test_behaviors)
            rnn.train()
            
            print (f'Epoch [{epoch+1}/{num_epochs}], Loss: {train_cost.item():.4f}, Test Loss: {test_cost.item():.4f}, R2: {r2_score(test_behaviors, test_outputs).item():.4f}')
                    
    final_r2 = r2_score(test_behaviors, test_outputs).item()

    if 'lfads' in model_name:
        mn = 'lfads'
    elif 'band' in model_name:
        mn = 'band'
    else:
        raise ValueError(f'Unknown model name {model_name}')
    print(mn, ' final R2: ',final_r2)

    # save predictions
    results_path = f'./results/{short_dataset_name}.h5'
    with h5py.File(results_path, 'a') as f:
        save_results(f,area,mn,train_outputs.cpu().detach().numpy(), test_outputs.cpu().detach().numpy())