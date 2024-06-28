import os
import shutil
from datetime import datetime
from pathlib import Path

from lfads_torch.run_model import run_model

# ---------- OPTIONS -----------
PROJECT_STR = "tauriel"
DATASET_STR = "tauriel_SC_velocities"  
N = 116 # 136 RSC, 116 SC
RUN_TAG = datetime.now().strftime("%y%m%d_%H%M%S") + "_test"
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
run_model(
    overrides={
        "datamodule": DATASET_STR,
        "model": PROJECT_STR,
        "logger.wandb_logger.project": PROJECT_STR,
        "logger.wandb_logger.tags.1": DATASET_STR,
        "logger.wandb_logger.tags.2": RUN_TAG,
        "model.encod_data_dim": N,
    },
    config_path="../configs/single.yaml",
)