from omegaconf import OmegaConf

OmegaConf.register_new_resolver(
    "relpath", lambda p: str(Path('/disk/scratch2/nkudryas/BAND-torch/scripts/').parent / ".." / p)
) # this is required for hydra to find the config files

import matplotlib.pyplot as plt
# import numpy as np
# import h5py
# from sklearn.linear_model import Ridge 
# from sklearn.decomposition import PCA
import torch
from hydra.utils import instantiate
from pathlib import Path
import hydra


from lfads_torch.utils import flatten


dataset_name = 'chewie_10_07'
PATH = 'f"/disk/scratch2/nkudryas/BAND-torch/datasets'

bin_size_ms = 10
best_model_dest = f"/disk/scratch2/nkudryas/BAND-torch/runs/band-torch-kl/{dataset_name}/"
model_name = '240119_154006_kl'
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