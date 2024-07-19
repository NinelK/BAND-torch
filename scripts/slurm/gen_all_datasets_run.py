#!/usr/bin/env python3
"""Script for generating experiments.txt"""
import os

# The home dir on the node's scratch disk
USER = os.getenv('USER')
# This may need changing to e.g. /disk/scratch_fast depending on the cluster
SCRATCH_DISK = '/disk/scratch'  
SCRATCH_HOME = f'{SCRATCH_DISK}/{USER}'

DATA_HOME = f'{SCRATCH_HOME}/BAND_torch/datasets'

model='kl1_gauss_bd'
fac_dim=100
co_dim=4

n_all = {'mihili_02_03': 114, 'mihili_02_17': 148, 'mihili_02_18': 159, 'mihili_03_07': 92, 'chewie_09_15': 309, 'chewie_09_21': 303, 'chewie_10_05': 244, 'chewie_10_07': 207}
n_m1 = {'mihili_02_03': 34, 'mihili_02_17': 44, 'mihili_02_18': 38, 'mihili_03_07': 26, 'chewie_09_15': 76, 'chewie_09_21': 72, 'chewie_10_05': 81, 'chewie_10_07': 70}
n_pmd = {'mihili_02_03': 80, 'mihili_02_17': 104, 'mihili_02_18': 121, 'mihili_03_07': 66, 'chewie_09_15': 233, 'chewie_09_21': 231, 'chewie_10_05': 163, 'chewie_10_07': 137}
T = {'mihili_02_03': 92, 'mihili_02_17': 85, 'mihili_02_18': 86, 'mihili_03_07': 87, 'chewie_09_15': 102, 'chewie_09_21': 103, 'chewie_10_05': 100, 'chewie_10_07': 101}

folds = range(5)

settings = folds
nr_expts = len(folds) * len(n_all) * 3

nr_servers = 15
avg_expt_time = 3*60  # mins
print(f'Total experiments = {nr_expts}')
print(f'Estimated time = {(nr_expts / nr_servers * avg_expt_time)/60} hrs')

output_file = open("scripts/slurm/experiment_across_datasets.txt", "w")

for dataset in n_all:
  for model_name, bw in zip(['lfads','band'],[0,0.1]):
    print(f'python scripts/run_pbt_slurm.py {model} {dataset} {model_name}_both_{fac_dim}f_{model} {T[dataset]} {fac_dim} {co_dim} {n_all[dataset]} {bw}', file=output_file)
    print(f'python scripts/run_pbt_slurm.py {model} {dataset}_M1 {model_name}_M1_{fac_dim}f_{model} {T[dataset]} {fac_dim} {co_dim} {n_m1[dataset]} {bw}', file=output_file)
    print(f'python scripts/run_pbt_slurm.py {model} {dataset}_PMd {model_name}_PMd_{fac_dim}f_{model} {T[dataset]} {fac_dim} {co_dim} {n_pmd[dataset]} {bw}', file=output_file)

output_file.close()
