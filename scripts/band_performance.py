from omegaconf import OmegaConf

OmegaConf.register_new_resolver(
    "relpath", lambda p: str(Path('/disk/scratch2/nkudryas/BAND-torch/scripts/').parent / ".." / p)
) # this is required for hydra to find the config files

import matplotlib.pyplot as plt
from matplotlib import cm

import numpy as np
import h5py
from sklearn.linear_model import Ridge 
from sklearn.decomposition import PCA
import torch
from hydra.utils import instantiate
from pathlib import Path
import hydra

from lfads_torch.utils import flatten

def R2(pred_beh,true_beh):
    return (1 - np.sum((pred_beh - true_beh)**2) / np.sum((true_beh - np.mean(true_beh))**2)) * 100

def get_target_ids(true_target_direction):
    ''' substitute direction elements with ids '''
    uniq_dirs = np.unique(true_target_direction)
    true_label = np.array([np.where(uniq_dirs==t)[0][0] for t in true_target_direction])
    return true_label

def plot_avg_traj(data,true_target_direction,title='',epoch_mask=True):
    ''' plot average trajectory for each target direction '''
    true_label = get_target_ids(true_target_direction)
    n = data.shape[-1]
    col, row = min(n,5), max(1,n//5)
    fig, ax = plt.subplots(row,col, figsize=(col*3,row*2),sharex=True)
    ax = ax.flatten()
    for i in range(n):
        for j in np.unique(true_label):
            ax[i].plot(data[(true_label==j) & epoch_mask].mean(0)[...,i],label=f'{true_target_direction[true_label==j][0]:.2f}')
        ax[i].set_title(f'{title} {i}')
        if i==(col-1):
            ax[i].legend(loc = (1.01,0))
    fig.tight_layout()
    return fig

dataset_name = 'chewie_10_07'
bin_width_sec = 0.01 # chewie
PATH = 'f"/disk/scratch2/nkudryas/BAND-torch/datasets'

best_model_dest = f"/disk/scratch2/nkudryas/BAND-torch/runs/band-torch-kl/{dataset_name}/"
# import glob
# for model_dest in glob.glob(f"{best_model_dest}/*")[::-1]:
model_name = '240201_134408_band_40f_kl1_student'
model_dest = f"{best_model_dest}/{model_name}"

# Load model
overrides={
        "datamodule": dataset_name,
        "model": dataset_name
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

# load the dataset
dataset_filename = '/disk/scratch2/nkudryas/BAND-torch/datasets/Chewie_CO_FF_2016-10-07_session_vel_M1_spikes_go.h5'

with h5py.File(dataset_filename, 'r') as f:
    train_data = f['train_recon_data'][:]
    valid_data = f['valid_recon_data'][:]
    train_inds, valid_inds = f["train_inds"][:], f["valid_inds"][:]
    valid_epoch = f["valid_epoch"][:]
    true_train_beh = f['train_vel'][:]
    true_valid_beh = f['valid_vel'][:]
    true_target_direction = f['valid_target_direction'][:]

# load model components
data_path = best_model_dest + model_name + '/lfads_output_sess0.h5'
with h5py.File(data_path) as f:
    # print(f.keys())
    # Merge train and valid data for factors and rates
    train_inds, valid_inds = f["train_inds"][:], f["valid_inds"][:]
    factors = f["valid_factors"][:]
    rates = f["valid_output_params"][:] / bin_width_sec
    behavior = f["valid_output_behavior_params"][:]
    controls = f['valid_gen_inputs'][:]
    ic = f['valid_gen_init'][:]

    train_factors = f["train_factors"][:]
    train_controls = f['train_gen_inputs'][:]
    train_ic = f['train_gen_init'][:]

# load ablated model components
data_path = best_model_dest + model_name + '/lfads_ablated_output_sess0.h5'
with h5py.File(data_path) as f:
    noci_factors = f["valid_factors"][:]
    noci_behavior = f["valid_output_behavior_params"][:]
    # noci_controls = f['valid_gen_inputs'][:]
    
# Run behavior prediction
# train Ridge regression to predict behavior from factors (0lag)
X_train = train_factors.reshape(-1,train_factors.shape[-1])
Y_train = true_train_beh.reshape(-1,true_train_beh.shape[-1])
X_test = factors.reshape(-1,factors.shape[-1])
ridge = Ridge(alpha=1).fit(X_train, Y_train)
Y_pred_0lag = ridge.predict(X_test).reshape(true_valid_beh.shape)
Y_pred_noci_0lag = ridge.predict(noci_factors.reshape(-1,noci_factors.shape[-1])).reshape(true_valid_beh.shape)

# Ridge from controls (0lag)
X_train = train_controls.reshape(-1,train_controls.shape[-1])
Y_train = true_train_beh.reshape(-1,true_train_beh.shape[-1])
X_test = controls.reshape(-1,controls.shape[-1])
ridge = Ridge(alpha=1).fit(X_train, Y_train)
Y_pred_control_0lag = ridge.predict(X_test).reshape(true_valid_beh.shape)

# Ridge seq2seq
X_train = train_factors.reshape(train_factors.shape[0],-1)
Y_train = true_train_beh.reshape(true_train_beh.shape[0],-1)
X_test = factors.reshape(factors.shape[0],-1)
ridge = Ridge(alpha=1).fit(X_train, Y_train)
Y_pred_seq2seq = ridge.predict(X_test).reshape(true_valid_beh.shape)
Y_pred_noci_seq2seq = ridge.predict(noci_factors.reshape(noci_factors.shape[0],-1)).reshape(true_valid_beh.shape)

# Ridge from control inputs (seq2seq)
X_train = train_controls.reshape(train_controls.shape[0],-1)
Y_train = true_train_beh.reshape(true_train_beh.shape[0],-1)
X_test = controls.reshape(controls.shape[0],-1)
ridge = Ridge(alpha=1).fit(X_train, Y_train)
Y_pred_control = ridge.predict(X_test).reshape(true_valid_beh.shape)

# Plot 1: plot behavior weight matrices
seq_len = config.model.recon_seq_len
in_features = config.model.behavior_readout.modules[0].in_features
out_features = config.model.behavior_readout.modules[0].out_features
beh_W = model.behavior_readout[0].layers[1].weight.T

assert beh_W.shape == (in_features*seq_len, out_features*seq_len)

beh_W = beh_W.reshape((seq_len, in_features, seq_len, out_features))

r = torch.std(beh_W)*4
fig, ax = plt.subplots(out_features, in_features, figsize=(in_features, out_features))
for j in range(in_features):
    for i in range(out_features):
        ax[i,j].imshow(beh_W[:,j,:,i].detach().numpy(), cmap='RdBu', vmin=-r, vmax=r)
        ax[i,j].set_xticks([])
        ax[i,j].set_yticks([])

fig.savefig(f"{model_dest}/behavior_weights.png")

# Plot 2: plot ICs

# PCA on initial conditions
pca = PCA(n_components=2)
pca.fit(train_ic)
ic_pca = pca.transform(ic)
# print(train_ic.shape,ic_pca.shape)

fig = plt.figure(figsize=(2.5,2))
plt.title('ICs')
plt.scatter(*ic_pca.T,c=cm.rainbow(get_target_ids(true_target_direction)/8),s=2)

fig.savefig(f"{model_dest}/initial_conditions.png")

# Plot 3: plot factors / controls / behavior prediction for 1 example trial

trial_id = 13
fig, ax = plt.subplots(3,4, figsize=(15,6),sharex=True)
ax[0,0].plot(factors[trial_id] - factors[trial_id].mean(0))
ax[1,0].plot(controls[trial_id])
ax[2,0].plot(noci_factors[trial_id])
ax[0,0].set_title('factors')
ax[1,0].set_title('controls')

c = ['C0','C1']
for i in range(2):
    ax[0,1].plot(Y_pred_0lag[trial_id][:,i],c=c[i])
    ax[1,1].plot(Y_pred_control_0lag[trial_id][:,i],c=c[i])
    ax[2,1].plot(Y_pred_noci_0lag[trial_id][:,i],c=c[i])

    ax[0,2].plot(Y_pred_seq2seq[trial_id][:,i],c=c[i])
    ax[1,2].plot(Y_pred_control[trial_id][:,i],c=c[i])
    ax[2,2].plot(Y_pred_noci_seq2seq[trial_id][:,i],c=c[i])
    
    ax[0,3].plot(behavior[trial_id][:,i],c=c[i])
    ax[2,3].plot(noci_behavior[trial_id][:,i],c=c[i])
    
    for j in range(ax.shape[0]):
        for k in range(1,ax.shape[1]):
            ax[j,k].plot(true_valid_beh[trial_id][:,i],c=c[i],linestyle='--')

ax[0,1].set_title(f'0lag from factors (R2 = {R2(Y_pred_0lag,true_valid_beh):.1f}%)')
ax[1,1].set_title(f'0lag from controls (R2 = {R2(Y_pred_control_0lag,true_valid_beh):.1f}%)')
ax[2,1].set_title(f'0lag from factors no CI (R2 = {R2(Y_pred_noci_0lag,true_valid_beh):.1f}%)')
ax[0,2].set_title(f'seq2seq from factors (R2 = {R2(Y_pred_seq2seq,true_valid_beh):.1f}%)')
ax[1,2].set_title(f'seq2seq from controls (R2 = {R2(Y_pred_control,true_valid_beh):.1f}%)')
ax[2,2].set_title(f'seq2seq from factors no CI (R2 = {R2(Y_pred_noci_seq2seq,true_valid_beh):.1f}%)')

ax[0,3].set_title(f'band behavior (R2 = {R2(behavior,true_valid_beh):.1f}%)')
ax[2,3].set_title(f'band behavior no CI (R2 = {R2(noci_behavior,true_valid_beh):.1f}%)')
fig.tight_layout()

fig.savefig(f"{model_dest}/factors_controls_behavior.png")

# Plot 4: plot avg factors and controls per condition (BL / AD / WO)
for epoch, epoch_name in enumerate(['BL','AD','WO']):
    fig = plot_avg_traj(factors,true_target_direction,title='factor',epoch_mask=(valid_epoch == epoch))
    fig.savefig(f"{model_dest}/avg_factors_{epoch_name}.png")
    fig = plot_avg_traj(noci_factors,true_target_direction,title='factor with no CI',epoch_mask=(valid_epoch == epoch))
    fig.savefig(f"{model_dest}/avg_noci_factors_{epoch_name}.png")
    fig = plot_avg_traj(controls,true_target_direction,title='control',epoch_mask=(valid_epoch == epoch))
    fig.savefig(f"{model_dest}/avg_controls_{epoch_name}.png")

fig = plot_avg_traj(factors,true_target_direction,title='factor')
fig.savefig(f"{model_dest}/avg_factors.png")
fig = plot_avg_traj(noci_factors,true_target_direction,title='factor with no CI')
fig.savefig(f"{model_dest}/avg_noci_factors.png")
fig = plot_avg_traj(controls,true_target_direction,title='control')
fig.savefig(f"{model_dest}/avg_controls.png")