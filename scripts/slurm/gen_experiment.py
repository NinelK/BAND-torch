#!/usr/bin/env python3
"""Script for generating experiments.txt"""
import os

# The home dir on the node's scratch disk
USER = os.getenv('USER')
# This may need changing to e.g. /disk/scratch_fast depending on the cluster
SCRATCH_DISK = '/disk/scratch'  
SCRATCH_HOME = f'{SCRATCH_DISK}/{USER}'

DATA_HOME = f'{SCRATCH_HOME}/BAND_torch/datasets'

model_name='band'
model='kl1_gauss_bd_d20_causal_ci'
fac_dim=100
co_dim=4

dataset='chewie_10_07_mov'
n_all=207
n_m1=70
n_pmd=137
T=101

#call = lambda fold: f'python scripts/run_pbt_slurm.py {model} {dataset}_cv{fold} {model_name}_both_{fac_dim}f_{co_dim}c_{model} {T} {fac_dim} {co_dim} {n_all}'
#call = lambda fold: f'python scripts/run_autolfads_pbt_slurm.py {model} {dataset}_cv{fold} {model_name}_both_{fac_dim}f_{co_dim}c_{model} {T} {fac_dim} {co_dim} {n_all}'
call = lambda fold: f'python scripts/run_single_slurm.py {model} {dataset}_cv{fold} {model_name}_both_{fac_dim}f_{co_dim}c_{model} {T} {fac_dim} {co_dim} {n_all}'

folds = range(5)

settings = folds
nr_expts = len(folds)

nr_servers = 5
avg_expt_time = 4*60  # mins
print(f'Total experiments = {nr_expts}')
print(f'Estimated time = {(nr_expts / nr_servers * avg_expt_time)/60} hrs')

output_file = open("./scripts/slurm/experiment.txt", "w")

for fold in settings:   
    # Note that we don't set a seed for rep - a seed is selected at random
    # and recorded in the output data by the python script
    expt_call = call(fold)
    print(expt_call, file=output_file)

output_file.close()
