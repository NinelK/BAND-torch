import shutil
from datetime import datetime
from pathlib import Path

from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import FIFOScheduler
from ray.tune.suggest.basic_variant import BasicVariantGenerator

from lfads_torch.run_model import run_model

# ---------- OPTIONS -----------
PROJECT_STR = "pbt"
DATASET_STR = "longitudinal_data1_multisession"
fac_dim = 10
co_dim = 0
cpus = 6

postfix = f'{fac_dim}f_{co_dim}c'

for bw, model_name in zip([0.0,0.1],['lfads','band']):
    RUN_TAG = f"{model_name}_{postfix}"
    RUN_DIR = Path("/disk/scratch/nkudryas/BAND-torch/runs") / PROJECT_STR / DATASET_STR / RUN_TAG

    # Set the mandatory config overrides to select datamodule and model
    mandatory_overrides = {
        "datamodule": DATASET_STR,
        "model": DATASET_STR,
        "logger.wandb_logger.project": PROJECT_STR,
        "logger.wandb_logger.tags.1": DATASET_STR,
        "logger.wandb_logger.tags.2": RUN_TAG,
        "model.fac_dim": fac_dim,
        "model.co_dim": co_dim,
        "model.behavior_weight": bw,
    }
    RUN_DIR.mkdir(parents=True)
    # Copy this script into the run directory
    shutil.copyfile(__file__, RUN_DIR / Path(__file__).name)
    # Run the hyperparameter search
    tune.run(
        tune.with_parameters(
            run_model,
            config_path="../configs/multi.yaml",
        ),
        metric="valid/recon_smth",
        mode="min",
        name=RUN_DIR.name,
        config={
            **mandatory_overrides,
            "model.dropout_rate": tune.uniform(0.0, 0.6),
            "model.l2_gen_scale": tune.loguniform(1e-4, 1e0),
            "model.l2_con_scale": tune.loguniform(1e-4, 1e0),
        },
        resources_per_trial=dict(cpu=3, gpu=0.5),
        num_samples=60,
        local_dir=RUN_DIR.parent,
        search_alg=BasicVariantGenerator(random_state=0),
        scheduler=FIFOScheduler(),
        verbose=1,
        progress_reporter=CLIReporter(
            metric_columns=["valid/recon_smth", "cur_epoch"],
            sort_by_metric=True,
        ),
        trial_dirname_creator=lambda trial: str(trial),
    )
