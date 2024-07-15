import os
import shutil
from glob import glob
# from datetime import datetime
import sys
from pathlib import Path

from ray import tune
from ray.tune import CLIReporter
from ray.tune.search.basic_variant import BasicVariantGenerator

from lfads_torch.extensions.tune import (
    BinaryTournamentPBT,
    HyperParam,
    ImprovementRatioStopper,
)
from lfads_torch.run_model import run_model

# ---------- OPTIONS ----------
PROJECT_STR = "pbt-band-paper"
MODEL_STR = sys.argv[1]
DATASET_STR = sys.argv[2]
RUN_TAG = sys.argv[3]
encod_seq_len = sys.argv[4]
fac_dim = sys.argv[5]
co_dim = sys.argv[6]
cpus = 2

RUN_DIR = Path("/disk/scratch/nkudryas/BAND-torch/runs") / PROJECT_STR / DATASET_STR / RUN_TAG

fold = None
if '_cv' in DATASET_STR:
    DATASET_STR, fold = DATASET_STR.split('_cv')
    print('CV fold: ',fold)

HYPERPARAM_SPACE = {
    "model.lr_init": HyperParam(
        1e-5, 5e-3, explore_wt=0.3, enforce_limits=True, init=4e-3
    ),
    "model.dropout_rate": HyperParam(
        0.0, 0.6, explore_wt=0.3, enforce_limits=True, sample_fn="uniform"
    ),
    "model.train_aug_stack.transforms.0.cd_rate": HyperParam(
        0.01, 0.7, explore_wt=0.3, enforce_limits=True, init=0.5, sample_fn="uniform"
    ),
    "model.behavior_weight": HyperParam(1e-2, 1e-1, explore_wt=0.3),
}
# ------------------------------


# Function to keep dropout and CD rates in-bounds
def clip_config_rates(config):
    return {k: min(v, 0.99) if "_rate" in k else v for k, v in config.items()}


init_space = {name: tune.sample_from(hp.init) for name, hp in HYPERPARAM_SPACE.items()}
# Set the mandatory config overrides to select datamodule and model
mandatory_overrides = {
    "datamodule": DATASET_STR,
    "model": MODEL_STR,
    # "logger.wandb_logger.project": PROJECT_STR,
    # "logger.wandb_logger.tags.1": DATASET_STR,
    # "logger.wandb_logger.tags.2": RUN_TAG,
    "model.encod_seq_len": encod_seq_len,
    "model.recon_seq_len": encod_seq_len,
    "model.kl_co_scale": float(encod_seq_len),
    "model.fac_dim": fac_dim,
    "model.co_dim": co_dim,
    "model.encod_data_dim": sys.argv[7],
    # "model.behavior_weight": sys.argv[8],
}
if fold is not None:
    mandatory_overrides["datamodule.fold"] = fold

RUN_DIR.mkdir(parents=True)
# Copy this script into the run directory
shutil.copyfile(__file__, RUN_DIR / Path(__file__).name)
# Run the hyperparameter search
# metric = "valid/recon_smth"
metric = "valid/pbt_target"
num_trials = 20
perturbation_interval = 25
burn_in_period = 80 + 25
analysis = tune.run(
    tune.with_parameters(
        run_model,
        config_path="../configs/pbt.yaml",
        do_posterior_sample=False,
    ),
    metric=metric,
    mode="min",
    name=RUN_DIR.name,
    stop=ImprovementRatioStopper(
        num_trials=num_trials,
        perturbation_interval=perturbation_interval,
        burn_in_period=burn_in_period,
        metric=metric,
        patience=4,
        min_improvement_ratio=5e-4,
    ),
    config={**mandatory_overrides, **init_space},
    resources_per_trial=dict(cpu=cpus, gpu=0.5),
    num_samples=num_trials,
    local_dir=RUN_DIR.parent,
    search_alg=BasicVariantGenerator(random_state=0),
    scheduler=BinaryTournamentPBT(
        perturbation_interval=perturbation_interval,
        burn_in_period=burn_in_period,
        hyperparam_mutations=HYPERPARAM_SPACE,
    ),
    keep_checkpoints_num=1,
    verbose=1,
    progress_reporter=CLIReporter(
        metric_columns=[metric, "cur_epoch"],
        sort_by_metric=True,
    ),
    trial_dirname_creator=lambda trial: str(trial),
)
# Copy the best model to a new folder so it is easy to identify
best_model_dir = RUN_DIR / "best_model"
shutil.copytree(analysis.best_logdir, best_model_dir)
# Switch working directory to this folder (usually handled by tune)
os.chdir(best_model_dir)
# Load the best model and run posterior sampling (skip training)
best_ckpt_dir = best_model_dir / Path(analysis.best_checkpoint._local_path).name
print(best_ckpt_dir)
run_model(
    overrides=mandatory_overrides,
    checkpoint_dir=best_ckpt_dir,
    config_path="../configs/pbt.yaml",
    do_train=False,
)

os.chdir('/home/nkudryas/experiments/BAND-torch/')

cmd = f'python scripts/ablate_controls.py {PROJECT_STR}'
for arg in sys.argv:
    cmd += ' ' + arg
os.system(cmd)

# cmd = f'python scripts/band_performance.py {PROJECT_STR}'
# for arg in sys.argv:
#     cmd += ' ' + arg
# os.system(cmd)

# # if need to re-sample
# # Copy the best model to a new folder so it is easy to identify
# best_model_dir = RUN_DIR / "best_model"
# # Switch working directory to this folder (usually handled by tune)
# os.chdir(best_model_dir)
# # Load the best model and run posterior sampling (skip training)
# checkpoint_folders = glob(str(best_model_dir)+'/checkpoint*')
# best_ckpt_dir = checkpoint_folders[-1] #+ '/tune.ckpt'
# print(best_ckpt_dir)
# run_model(
#     overrides=mandatory_overrides,
#     checkpoint_dir=best_ckpt_dir,
#     config_path="../configs/pbt.yaml",
#     do_train=False,
# )

