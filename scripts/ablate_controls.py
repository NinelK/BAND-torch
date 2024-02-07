from lfads_torch.post_run.band_analysis import run_posterior_sampling
import torch
from hydra.utils import instantiate
from pathlib import Path
import hydra
import os

from lfads_torch.utils import flatten

from omegaconf import OmegaConf

parent_path = '/disk/scratch2/nkudryas/BAND-torch'

OmegaConf.register_new_resolver(
    "relpath", lambda p: str(Path(f'{parent_path}/scripts/').parent / p)
)

dataset_name = 'chewie_10_07'
PATH = parent_path + '/datasets'

best_model_dest = f"{parent_path}/runs/band-torch-kl/{dataset_name}"

model_name = '240201_134408_band_40f_kl1_student'
model_dest = f"{best_model_dest}/{model_name}"

overrides={
        "datamodule": dataset_name,
        "model": dataset_name
    }
config_path="../configs/single.yaml"
print(config_path)

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


model.eval()
# zero out weight
B = torch.zeros_like(model.decoder.rnn.cell.co_linear.bias)
B[len(B) // 2:] = -10 # making variance exp(-10)
model.decoder.rnn.cell.co_linear.weight = torch.nn.Parameter(torch.zeros_like(model.decoder.rnn.cell.co_linear.weight))
model.decoder.rnn.cell.co_linear.bias = torch.nn.Parameter(B)

filename = 'lfads_ablated_output.h5' # if model_dest + '*.h5' -- still puts in the same directory, I DON'T KNOW WHY
run_posterior_sampling(model, datamodule, filename, num_samples=50)

# placing the output file in the right folder, assuming recording had a single session
filename = filename.split('.')[0] + '_sess0.h5'
os.replace(parent_path + '/' + filename, model_dest + '/' + filename)