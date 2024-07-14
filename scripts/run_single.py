import os
import shutil
from datetime import datetime
from pathlib import Path
import sys

from lfads_torch.run_model import run_model

# ---------- OPTIONS -----------
PROJECT_STR = "band-paper"
MODEL_STR = sys.argv[1]
DATASET_STR = sys.argv[2]
RUN_TAG = sys.argv[3]
encod_seq_len = sys.argv[4]
fac_dim = sys.argv[5]
co_dim = sys.argv[6]

RUN_DIR = Path("./runs") / PROJECT_STR / DATASET_STR / RUN_TAG
OVERWRITE = True

fold = None
if '_cv' in DATASET_STR:
    DATASET_STR, fold = DATASET_STR.split('_cv')
    print('CV fold: ',fold)

# ------------------------------

# Overwrite the directory if necessary
if RUN_DIR.exists() and OVERWRITE:
    shutil.rmtree(RUN_DIR)
RUN_DIR.mkdir(parents=True)
# Copy this script into the run directory
shutil.copyfile(__file__, RUN_DIR / Path(__file__).name)
# Switch to the `RUN_DIR` and train the model
os.chdir(RUN_DIR)
model_name = DATASET_STR.replace('_M1', '').replace('_PMd','')
datamodule = DATASET_STR.replace('_small','')

overrides={
        "datamodule": DATASET_STR,
        "model": MODEL_STR,
        "logger.wandb_logger.project": PROJECT_STR,
        "logger.wandb_logger.tags.1": DATASET_STR,
        "logger.wandb_logger.tags.2": RUN_TAG,
        "model.encod_seq_len": encod_seq_len,
        "model.recon_seq_len": encod_seq_len,
        "model.kl_co_scale": float(encod_seq_len),
        "model.fac_dim": fac_dim,
        "model.co_dim": co_dim,
        "model.encod_data_dim": sys.argv[7],
        "model.behavior_weight": sys.argv[8],
        
    }

if fold is not None:
    overrides["datamodule.fold"] = fold

run_model(
    overrides=overrides,
    config_path="../configs/single.yaml",
)
