import os
import shutil
from datetime import datetime
from pathlib import Path

from band_torch.run_model import run_model

# ---------- OPTIONS -----------
PROJECT_STR = "band-torch-example"
DATASET_STR = "nlb_area2_bump"
RUN_TAG = datetime.now().strftime("%y%m%d") + "_exampleSingle"
RUN_DIR = Path("../runs") / PROJECT_STR / DATASET_STR / RUN_TAG
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
    config_path="../configs/single.yaml",
)
