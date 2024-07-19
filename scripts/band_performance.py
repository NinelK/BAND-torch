from omegaconf import OmegaConf

OmegaConf.register_new_resolver(
    "relpath", lambda p: str(Path('/disk/scratch/nkudryas/BAND-torch/scripts/').parent / p)
) # this is required for hydra to find the config files

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.axes import Axes

import numpy as np
import pandas as pd
import h5py
from scipy.stats import poisson
from sklearn.linear_model import Ridge 
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import torch
from hydra.utils import instantiate
from pathlib import Path
import hydra
import sys

from lfads_torch.band_utils import flatten

name_translation = {
    'chewie_09_15': 'Chewie_CO_FF_2016-09-15',
    'chewie_09_21': 'Chewie_CO_FF_2016-09-21',
    'chewie_10_05': 'Chewie_CO_FF_2016-10-05',
    'chewie_10_07': 'Chewie_CO_FF_2016-10-07',
    'mihili_02_03': 'Mihili_CO_FF_2014-02-03',
    'mihili_02_17': 'Mihili_CO_FF_2014-02-17',
    'mihili_02_18': 'Mihili_CO_FF_2014-02-18',
    'mihili_03_07': 'Mihili_CO_FF_2014-03-07',
}

# TODO ideally this needs modularization. There are multiple versions of similar R2s and plots, 
# it should all be moved to lfads_torch/metrics.py and plot_helpers.py (which is now in notebooks/paper/plot_helpers.py)

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
    col, row = min(n,5), max(1,int(np.ceil(n/5)))
    fig, ax = plt.subplots(row,col, figsize=(col*3,row*2),sharex=True)
    if type(ax) is Axes:
        ax = [ax]
    else:
        ax = ax.flatten()
    assert len(ax)>=n, f'not enough axes created for {n} factors'
    for i in range(n):
        for j in np.unique(true_label):
            ax[i].plot(data[(true_label==j) & epoch_mask].mean(0)[...,i],label=f'{true_target_direction[true_label==j][0]:.2f}')
        ax[i].set_title(f'{title} {i}')
        if i==(col-1):
            ax[i].legend(loc = (1.01,0))
    for i in range(n,len(ax)):
        ax[i].axis("off")
    fig.tight_layout()
    return fig

def get_trials2plot(pos, pred_pos, dir_index, epoch, epoch2plot=1):
    trials2plot = np.zeros_like(epoch)
    for e in np.unique(epoch):
        for d in np.unique(dir_index):
            mask = (epoch == epoch2plot) & (dir_index == d)
            # print(mask)
            dist = ((pos - pred_pos) ** 2).sum(-1).sum(-1)
            dist[~mask] = -np.inf
            # print(dist)
            idx_max = np.argmax(dist)
            # print(idx_max)
            trials2plot[idx_max] = 1
    return trials2plot

def plot_beh_pred(vel, pred_vel, dir_index, trials2plot, file_name=""):
    pos = np.cumsum(vel * 0.01, 1)
    pred_pos = np.cumsum(pred_vel * 0.01, 1)

    fig = plt.figure(figsize=(6, 3))

    axes = [
        fig.add_axes([0, 0, 0.5, 1])
    ]

    ax_vel = [
        [fig.add_axes([0.50, 0.1 * i, 0.25, 0.1]) for i in range(dir_index.max()+1)],
        [fig.add_axes([0.75, 0.1 * i, 0.25, 0.1]) for i in range(dir_index.max()+1)],
    ]

    time = np.arange(pos.shape[1]) * 10

    for p, v, ls in zip([pos, pred_pos], [vel, pred_vel], [":", "solid"]):
        for t in range(0, pos.shape[0]):
            # ls = ':' if epoch[t]==0 else 'solid'
            if trials2plot[t]:
                axes[0].plot(
                    p[t, :, 0],
                    p[t, :, 1],
                    color=f"C{dir_index[t]}",
                    alpha=1,
                    ls=ls,
                )
                d = dir_index[t]
                for i in range(2):
                    ax_vel[i][d].plot(
                        time,
                        v[t, :, i],
                        color=f"C{d}",
                        alpha=1,
                        ls=ls,
                    )

    for ax in axes:
        ax.axis("off")

    for ax in ax_vel:
        for a in ax:
            a.axis("off")

    R2_iso_pos = 1 - np.sum((pos - pred_pos) ** 2) / np.sum((pos - pos.mean()) ** 2)
    R2_iso_vel = 1 - np.sum((vel - pred_vel) ** 2) / np.sum((vel - vel.mean()) ** 2)
    
    axes[0].text(np.min(pos[...,0]),np.max(pos[...,1]),f'R2_pos = {R2_iso_pos*100:.2f}%')
    ax_vel[0][-1].set_title(f'R2_vel = {R2_iso_vel*100:.2f}%')

    plt.savefig(file_name)

OLD = False

PROJECT_STR = sys.argv[1]
MODEL_STR = sys.argv[2]
DATASET_STR = sys.argv[3] #'chewie_10_07'
bin_width_sec = 0.01 # chewie
PATH = 'f"/disk/scratch/nkudryas/BAND-torch/datasets'

best_model_dest = f"/disk/scratch/nkudryas/BAND-torch/runs/{PROJECT_STR}/{DATASET_STR}/"
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
        "model": MODEL_STR+("_old" if OLD else ""), #dataset_name.replace('_M1', '').replace('_PMd',''),
        "model.encod_seq_len": encod_seq_len,
        "model.recon_seq_len": encod_seq_len,
        "model.fac_dim": sys.argv[6],
        "model.co_dim": sys.argv[7],
        "model.encod_data_dim": sys.argv[8],
    }
if fold is not None:
    overrides["datamodule.fold"] = fold

if 'lfads' in RUN_TAG:
    overrides["model.behavior_weight"] = 0.
    print('Zeroed out behavior weight to emulate LFADS')

config_path="../configs/pbt.yaml"
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
    if OLD:
        session = 'sess0'
    else:
        session = dataset_filename.split("/")[-1].split("_")[-1].split(".")[0]
        if fold is not None:
            session = f"cv{fold}"
    
    print(dataset_filename)

    with h5py.File(dataset_filename, 'r') as f:
        train_data = f['train_recon_data'][:]
        valid_data = f['valid_recon_data'][:]
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
        # print(f.keys())
        # Merge train and valid data for factors and rates
        # train_inds, valid_inds = f['train_inds'][:], f['valid_inds'][:]
        factors = f['valid_factors'][:]
        rates = f["valid_output_params"][:] / bin_width_sec
        train_behavior = f["train_output_behavior_params"][:]
        behavior = f["valid_output_behavior_params"][:]
        controls = f['valid_gen_inputs'][:]
        ic = f['valid_gen_init'][:]

        train_factors = f["train_factors"][:]
        train_controls = f['train_gen_inputs'][:]
        train_ic = f['train_gen_init'][:]

    if co_dim > 0:
        # load ablated model components
        if 'pbt' in PROJECT_STR:
            data_path = best_model_dest + model_name + f'/best_model/lfads_ablated_output_{session}.h5'
        else:
            data_path = best_model_dest + model_name + f'/lfads_ablated_output_{session}.h5'
        with h5py.File(data_path) as f:
            noci_factors = f["valid_factors"][:]
            noci_behavior = f["valid_output_behavior_params"][:]
            noci_rates = f["valid_output_params"][:] / bin_width_sec
            # noci_controls = f['valid_gen_inputs'][:]
        
    # calculate LL diff
    valid_mean_count = valid_data.mean(0).mean(0) # to get Hz -> x100
    base_LL = poisson.logpmf(valid_data, valid_mean_count)
    n_sp = valid_data.sum()

    valid_LL = {'LL': poisson.logpmf(valid_data, bin_width_sec * rates), 
                'LL_noci': poisson.logpmf(valid_data, bin_width_sec * noci_rates),}
    valid_co_bps = {key: np.sum(valid_LL[key] - base_LL)/(n_sp * np.log(2)) for key in valid_LL.keys()}
    # add same AD
    valid_co_bps_AD = {key+'_AD': np.sum(valid_LL[key][valid_epoch == 1] - base_LL[valid_epoch == 1])/(n_sp * np.log(2)) for key in valid_LL.keys()}
    valid_co_bps.update(valid_co_bps_AD)
    print(f"LL diff: {valid_co_bps}")
    # save to csv
    df = pd.DataFrame(valid_co_bps, index=[0])
    df.to_csv(f"{model_dest}/LL_diff.csv")
    
    # Run behavior prediction
    # train Ridge regression to predict behavior from factors (0lag)
    X_train = train_factors.reshape(-1,train_factors.shape[-1])
    Y_train = true_train_beh.reshape(-1,true_train_beh.shape[-1])
    X_test = factors.reshape(-1,factors.shape[-1])
    ridge = Ridge(alpha=1).fit(X_train, Y_train)
    Y_pred_0lag = ridge.predict(X_test).reshape(true_valid_beh.shape)
    if co_dim > 0:
        Y_pred_noci_0lag = ridge.predict(noci_factors.reshape(-1,noci_factors.shape[-1])).reshape(true_valid_beh.shape)

    # Ridge seq2seq
    X_train = train_factors.reshape(train_factors.shape[0],-1)
    Y_train = true_train_beh.reshape(true_train_beh.shape[0],-1)
    X_test = factors.reshape(factors.shape[0],-1)
    ridge = Ridge(alpha=1).fit(X_train, Y_train)
    Y_pred_seq2seq_train = ridge.predict(X_train).reshape(true_train_beh.shape)
    Y_pred_seq2seq = ridge.predict(X_test).reshape(true_valid_beh.shape)
    if co_dim > 0:
        Y_pred_noci_seq2seq = ridge.predict(noci_factors.reshape(noci_factors.shape[0],-1)).reshape(true_valid_beh.shape)

    if co_dim > 0:
        # Ridge from controls (0lag)
        X_train = train_controls.reshape(-1,train_controls.shape[-1])
        Y_train = true_train_beh.reshape(-1,true_train_beh.shape[-1])
        X_test = controls.reshape(-1,controls.shape[-1])
        ridge = Ridge(alpha=1).fit(X_train, Y_train)
        Y_pred_control_0lag = ridge.predict(X_test).reshape(true_valid_beh.shape)
            
        # Ridge from control inputs (seq2seq)
        X_train = train_controls.reshape(train_controls.shape[0],-1)
        Y_train = true_train_beh.reshape(true_train_beh.shape[0],-1)
        X_test = controls.reshape(controls.shape[0],-1)
        ridge = Ridge(alpha=1).fit(X_train, Y_train)
        Y_pred_control = ridge.predict(X_test).reshape(true_valid_beh.shape)

    # save results in the summary file
    def save_results(f,area,mn,fac_dim,co_dim,train_outputs, test_outputs,sample=''):
        if (sample=='') | (sample==0) | (sample=='0'):
            sample_str = ''
        else:
            sample_str = f'_sample{sample}'
        key = f'train_{area}_{mn}_{fac_dim}f_{co_dim}c{sample_str}_pred'
        if key in f:
            del f[key]
        key = f'test_{area}_{mn}_{fac_dim}f_{co_dim}c{sample_str}_pred'
        if key in f:
            del f[key]
        f.create_dataset(f'train_{area}_{mn}_{fac_dim}f_{co_dim}c{sample_str}_pred', data=train_outputs)
        f.create_dataset(f'test_{area}_{mn}_{fac_dim}f_{co_dim}c{sample_str}_pred', data=test_outputs)

    short_dataset_name = dataset_name.replace('_M1', '').replace('_PMd','').replace('_small','')
    if 'M1' in dataset_name:
        area = 'M1'
    elif 'PMd' in dataset_name:
        area = 'PMd'
    else:
        area = 'all'
    if 'lfads' in model_name:
        train_outputs = Y_pred_seq2seq_train
        test_outputs = Y_pred_seq2seq
        mn = 'lfads'
    elif 'band' in model_name:
        train_outputs = train_behavior
        test_outputs = behavior
        mn = 'band'
    else:
        raise ValueError(f'Unknown model name {model_name}')
    if short_dataset_name in name_translation:
        results_path = f'./results/{name_translation[short_dataset_name]}.h5'
        with h5py.File(results_path, 'a') as f:
            save_results(f,area,mn,factors.shape[-1],controls.shape[-1],train_outputs, test_outputs)
    else:
        print(f'Unknown dataset name {short_dataset_name}')

    # Plot 1: plot behavior weight matrices
    seq_len = config.model.recon_seq_len
    if OLD:
        in_features = config.model.behavior_readout.modules[0].in_features
        out_features = config.model.behavior_readout.modules[0].out_features
        beh_W = model.behavior_readout[0].layers[1].weight.T
    else:
        in_features = config.model.behavior_readout.in_features
        out_features = config.model.behavior_readout.out_features
        beh_W = model.behavior_readout.layers[1].weight.T

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

    # t-sne on initial conditions
    tsne = TSNE(n_components=2, perplexity=min(ic.shape[0]//2,30.0))
    ic_tsne = tsne.fit_transform(ic)

    fig, axes = plt.subplots(1,2,figsize=(5,2))
    fig.suptitle('ICs')
    target_ids = get_target_ids(true_target_direction)
    axes[0].scatter(*ic_pca.T,c=cm.rainbow(target_ids / target_ids.max()),s=2)
    axes[0].set_title('PCA')
    axes[0].set_xlabel('PC1')
    axes[0].set_ylabel('PC2')
    axes[1].scatter(*ic_tsne.T,c=cm.rainbow(target_ids / target_ids.max()),s=2)
    axes[1].set_title('t-SNE')
    axes[1].set_xlabel('t-SNE1')
    axes[1].set_ylabel('t-SNE2')

    # add colorbar
    sm = plt.cm.ScalarMappable(cmap=cm.rainbow, norm=plt.Normalize(vmin=0, vmax=target_ids.max()))
    sm._A = []
    fig.colorbar(sm, ax=axes, orientation='vertical')

    for ax in axes:
        ax.axis('equal')
        ax.set_xticks([])
        ax.set_yticks([])

    fig.savefig(f"{model_dest}/initial_conditions.png")

    # Plot 3: plot factors / controls / behavior prediction for 1 example trial

    trial_id = 0
    fig, ax = plt.subplots(3,4, figsize=(15,6),sharex=True)
    ax[0,0].plot(factors[trial_id] - factors[trial_id].mean(0))
    if co_dim > 0:
        ax[1,0].plot(controls[trial_id])
        ax[2,0].plot(noci_factors[trial_id])
    ax[0,0].set_title('factors')
    ax[1,0].set_title('controls')

    c = ['C0','C1']
    for i in range(2):
        ax[0,1].plot(Y_pred_0lag[trial_id][:,i],c=c[i])
        if co_dim > 0 :
            ax[1,1].plot(Y_pred_control_0lag[trial_id][:,i],c=c[i])
            ax[2,1].plot(Y_pred_noci_0lag[trial_id][:,i],c=c[i])

        ax[0,2].plot(Y_pred_seq2seq[trial_id][:,i],c=c[i])
        if co_dim > 0:
            ax[1,2].plot(Y_pred_control[trial_id][:,i],c=c[i])
            ax[2,2].plot(Y_pred_noci_seq2seq[trial_id][:,i],c=c[i])
        
        ax[0,3].plot(behavior[trial_id][:,i],c=c[i])
        if co_dim > 0:
            ax[2,3].plot(noci_behavior[trial_id][:,i],c=c[i])
        
        for j in range(ax.shape[0]):
            for k in range(1,ax.shape[1]):
                ax[j,k].plot(true_valid_beh[trial_id][:,i],c=c[i],linestyle='--')

    ax[0,1].set_title(f'0lag from factors (R2 = {R2(Y_pred_0lag,true_valid_beh):.1f}%)')
    ax[0,2].set_title(f'seq2seq from factors (R2 = {R2(Y_pred_seq2seq,true_valid_beh):.1f}%)')
    ax[0,3].set_title(f'band behavior (R2 = {R2(behavior,true_valid_beh):.1f}%)')
    if co_dim > 0:
        ax[1,1].set_title(f'0lag from controls (R2 = {R2(Y_pred_control_0lag,true_valid_beh):.1f}%)')
        ax[1,2].set_title(f'seq2seq from controls (R2 = {R2(Y_pred_control,true_valid_beh):.1f}%)')
        ax[2,1].set_title(f'0lag from factors no CI (R2 = {R2(Y_pred_noci_0lag,true_valid_beh):.1f}%)')
        ax[2,2].set_title(f'seq2seq from factors no CI (R2 = {R2(Y_pred_noci_seq2seq,true_valid_beh):.1f}%)')
        ax[2,3].set_title(f'band behavior no CI (R2 = {R2(noci_behavior,true_valid_beh):.1f}%)')

    # combine R2 results and save as csv
    R2_results = {  'seq2seq from factors': R2(Y_pred_seq2seq,true_valid_beh),
                    'seq2seq from factors no CI': R2(Y_pred_noci_seq2seq,true_valid_beh),
                    'seq2seq from factprs in AD': R2(Y_pred_seq2seq[valid_epoch == 1],true_valid_beh[valid_epoch == 1]),
                    'seq2seq from factors no CI in AD': R2(Y_pred_noci_seq2seq[valid_epoch == 1],true_valid_beh[valid_epoch == 1]),
                    'band behavior': R2(behavior,true_valid_beh),
                    'band behavior no CI': R2(noci_behavior,true_valid_beh),
                    'band behavior in AD': R2(behavior[valid_epoch == 1],true_valid_beh[valid_epoch == 1]),
                    'band behavior no CI in AD': R2(noci_behavior[valid_epoch == 1],true_valid_beh[valid_epoch == 1]),
                    'seq2seq from controls': R2(Y_pred_control,true_valid_beh)}
    # print(f"R2 results: {R2_results}")
    # save to csv

    df = pd.DataFrame(R2_results, index=[0])
    df.to_csv(f"{model_dest}/R2_results.csv")
    

    fig.tight_layout()

    fig.savefig(f"{model_dest}/factors_controls_behavior.png")

    # Plot 4: plot avg factors and controls per condition (BL / AD / WO)
    for epoch, epoch_name in enumerate(['BL','AD','WO']):
        fig = plot_avg_traj(factors,true_target_direction,title='factor',epoch_mask=(valid_epoch == epoch))
        fig.savefig(f"{model_dest}/avg_factors_{epoch_name}.png")
        if co_dim > 0:
            fig = plot_avg_traj(noci_factors,true_target_direction,title='factor with no CI',epoch_mask=(valid_epoch == epoch))
            fig.savefig(f"{model_dest}/avg_noci_factors_{epoch_name}.png")
            fig = plot_avg_traj(controls,true_target_direction,title='control',epoch_mask=(valid_epoch == epoch))
            fig.savefig(f"{model_dest}/avg_controls_{epoch_name}.png")

    fig = plot_avg_traj(factors,true_target_direction,title='factor')
    fig.savefig(f"{model_dest}/avg_factors.png")
    if co_dim > 0:
        fig = plot_avg_traj(noci_factors,true_target_direction,title='factor with no CI')
        fig.savefig(f"{model_dest}/avg_noci_factors.png")
        fig = plot_avg_traj(controls,true_target_direction,title='control')
        fig.savefig(f"{model_dest}/avg_controls.png")


    # Plot 5. Plot behavior prediction
    dir_index = np.array([sorted(set(true_target_direction)).index(i) for i in true_target_direction])
    avg_vel = np.empty_like(true_valid_beh)
    for d in range(np.max(dir_index) + 1):
        mask = d == dir_index
        avg_vel[mask] = true_valid_beh[mask].mean(0)
    if valid_epoch.max() >= 1:
        epoch2plot = 1 # AD
    else:
        epoch2plot = 0 # other datasets that don't have epochs
    trials2plot = get_trials2plot(true_valid_beh, avg_vel, dir_index, valid_epoch,epoch2plot=epoch2plot) # trials with max distance from avg vel
    plot_beh_pred(true_valid_beh, Y_pred_seq2seq, dir_index, trials2plot, f"{model_dest}/beh_prediction.png")
    plot_beh_pred(true_valid_beh, Y_pred_seq2seq, dir_index, trials2plot, f"{model_dest}/beh_prediction.svg")
    if co_dim > 0:
        plot_beh_pred(true_valid_beh, Y_pred_noci_seq2seq, dir_index, trials2plot, f"{model_dest}/beh_prediction_noci.png")
        plot_beh_pred(true_valid_beh, Y_pred_noci_seq2seq, dir_index, trials2plot, f"{model_dest}/beh_prediction_noci.svg")
