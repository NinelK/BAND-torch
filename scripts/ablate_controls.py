import torch
from hydra.utils import instantiate
from pathlib import Path
import hydra
import os
import sys
from glob import glob

from lfads_torch.band_utils import flatten
from lfads_torch.post_run.band_analysis import run_posterior_sampling

from omegaconf import OmegaConf

parent_path = '/disk/scratch2/nkudryas/BAND-torch'

OmegaConf.register_new_resolver(
    "relpath", lambda p: str(Path(f'{parent_path}/scripts/').parent / p)
)

PROJECT_STR = sys.argv[1]
MODEL_STR = sys.argv[2]
DATASET_STR = sys.argv[3]
PATH = parent_path + '/datasets'

best_model_dest = f"{parent_path}/runs/{PROJECT_STR}/{DATASET_STR}"

fold = None
if '_cv' in DATASET_STR:
    dataset_name, fold = DATASET_STR.split('_cv')
    print('CV fold: ',fold)
else:
    dataset_name = DATASET_STR

model_name = sys.argv[4]
model_dest = f"{best_model_dest}/{model_name}"

encod_seq_len = sys.argv[5]
overrides={
        "datamodule": dataset_name,
        "model": MODEL_STR, #dataset_name.replace('_M1', '').replace('_PMd',''),
        "model.encod_seq_len": encod_seq_len,
        "model.recon_seq_len": encod_seq_len,
        "model.fac_dim": sys.argv[6],
        "model.co_dim": sys.argv[7],
        "model.encod_data_dim": sys.argv[8],
    }
if fold is not None:
    overrides["datamodule.fold"] = fold

if 'lfads' in model_name:
    overrides["model.behavior_weight"] = 0.
    print('Zeroed out behavior weight to emulate LFADS')

config_path="../configs/pbt.yaml"
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

# Instantiate `Light    #ningDataModule` and `LightningModule`
datamodule = instantiate(config.datamodule, _convert_="all")
model = instantiate(config.model)

if 'pbt' in PROJECT_STR:
    # check the latest checkpoint
    checkpoint_folders = glob(model_dest+'/best_model/checkpoint*')
    ckpt_path = checkpoint_folders[-1] + '/tune.ckpt'
else:
    ckpt_path = f'{model_dest}/lightning_checkpoints/last.ckpt'
model.load_state_dict(torch.load(ckpt_path)["state_dict"])

model.eval()
# zero out weight
model.decoder.rnn.cell.gen_cell.weight_ih = torch.nn.Parameter(torch.zeros_like(model.decoder.rnn.cell.gen_cell.weight_ih))

filename_source = f'lfads_W_ablated_output_{model_name}.h5' # if model_dest + '*.h5' -- 
#TODO fix the issue that run_posterior_sampling still puts the file the same directory, ignoring the path, I DON'T KNOW WHY
# current workaround: give the file a unique name (to avoid clashes between parallel runs) and then copy -\__o_O__/-
data_path = datamodule.hparams.datafile_pattern
if fold is not None:    #TODO: figure out why override did not set it up properly?
    data_path = data_path.replace('.h5', f'_cv{fold}.h5')
data_paths = sorted(glob(data_path))
assert len(data_paths)>0, f'Nothing matches {data_path}'
# Give each session a unique file path
for s in range(len(data_paths)):
    session = data_paths[s].split("/")[-1].split("_")[-1].split(".")[0]
    if fold is not None:
        session = f'cv{fold}'
    filename = f'lfads_W_ablated_output_{session}.h5' # if model_dest + '*.h5'
    run_posterior_sampling(model, datamodule, filename_source, num_samples=50)

    # placing the output file in the right folder, assuming recording had a single session
    filename_source = filename_source.split('.')[0] + f'_{session}.h5'
    if 'pbt' in PROJECT_STR:
        os.replace(parent_path + '/' + filename_source, model_dest + '/best_model/' + filename)
    else:
        os.replace(parent_path + '/' + filename_source, model_dest + '/' + filename)
