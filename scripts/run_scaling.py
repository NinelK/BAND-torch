import os
import shutil
from datetime import datetime
from pathlib import Path
import sys

from lfads_torch.run_model import run_model

# ---------- OPTIONS -----------
PROJECT_STR = "band-paper"
DATASET_STR = sys.argv[1] #"chewie_10_07"  # "nlb_area2_bump"
RUN_TAG = sys.argv[2] #datetime.now().strftime("%y%m%d_%H%M%S") + "_kl"
RUN_DIR = Path("./runs") / PROJECT_STR / DATASET_STR / RUN_TAG
OVERWRITE = True
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
datamodule = DATASET_STR
run_model(
    overrides={
        "datamodule": datamodule,
        "model": model_name,
        "logger.wandb_logger.project": PROJECT_STR,
        "logger.wandb_logger.tags.1": DATASET_STR,
        "logger.wandb_logger.tags.2": RUN_TAG,
        "model.fac_dim": sys.argv[3],
        "model.co_dim": sys.argv[4],
        "model.encod_data_dim": sys.argv[5],
        "model.behavior_weight": sys.argv[6],
        
    },
    config_path="../configs/single.yaml",
)
