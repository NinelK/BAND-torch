# `BAND-torch`: Behavior-aligned neural dynamics model

The code in this repository extends `lfads-torch`: A modular and extensible implementation of latent factor analysis via dynamical systems
[![arXiv](https://img.shields.io/badge/arXiv-2309.01230-b31b1b.svg)](https://arxiv.org/abs/2309.01230)

BAND is a latent dynamics model weakly supervised with behavior.
Using a well-established latent dynamics model (LFADS) as a baseline, we constructed a model that not only explains neural variability but also aligns the latent space with the behavioral output ([BAND schema](BAND_schema.pdf), green is new).
While LFADS is capable of inferring inputs through the controller RNN, this is only the case when these inputs cause
a significant change in future dynamics and affect the neural reconstruction.
To ensure that behavior-related inputs can be captured even if they cause a small, transient change in neural dynamics, we utilize an additional behavior decoder (Fig.~\ref{fig:fig1}b, green).

Latent factor analysis via dynamical systems (LFADS) is a variational sequential autoencoder that achieves state-of-the-art performance in denoising high-dimensional neural spiking activity for downstream applications in science and engineering [1, 2, 3, 4]. Recently introduced variants have continued to demonstrate the applicability of the architecture to a wide variety of problems in neuroscience [5, 6, 7, 8]. Since the development of the original implementation of LFADS, new technologies have emerged that use dynamic computation graphs [9], minimize boilerplate code [10], compose model configuration files [11], and simplify large-scale training [12]. Building on these modern Python libraries, we introduce `band-torch` &mdash; a new open-source implementation of LFADS designed to be easier to understand, configure, and extend.

# Installation
To create an environment and install the dependencies of the project, run the following commands:
```
git clone https://github.com/NinelK/BAND-torch.git
cd BAND-torch
micromamba create --name band-torch python=3.9
micromamba activate band-torch
pip install -e .
pre-commit install
```

# Notes on fixing problems

To fix `/lib64/libstdc++.so.6: version `CXXABI_1.3.9'` error, add path to this library in your env, e.g.:
`export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/disk/scratch2/nkudryas/micromamba/envs/band-torch/lib/`

To fix `ImportError: cannot import name 'packaging' from 'pkg_resources'`:
downgrade setuptools <70:
`micromamba install setuptools==69.5.1`

To fix `AttributeError: module 'numpy' has no attribute 'bool8'`:
downgrade numpy < 2
`pip install numpy==1.26.0 scikit-learn==1.3.0`
basically, `import sklearn` won't work without these downgrades. Until sklearn can't be imported, BAND will throw confusing hydra-related errors.

Backward compatibility for newer PyTorch:
`export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`

# slurm notes

Follow quick [setup guidelines](https://docs.google.com/document/d/1C4x2Ne0lpm4KpfoIPH1_afcQVF_qoW8NWqyIIndUpow/edit)

create `~/experiments` folder (`~/` is a DFS home)

clone BAND-torch there

(if not the first time using slurm server, then just connect and start here:
`ssh ${USER}@mlp.inf.ed.ac.uk` )

pull `*.h5` datasets into `~/experiments/BAND-torch/datasets` (from whereever you have them, AFS or another server on the network)
e.g.
`scp bruichladdich:/disk/scratch2/nkudryas/BAND-torch/datasets/multi* ~/experiments/BAND-torch/datasets/`

also copy all the relevant configs for these datasets and the models you plan to run (in `~/experiments/BAND-torch/configs`)
  (either on the head node, since only that one has internet on MLP, or in an interactive session)
`scp -r bruichladdich:/disk/scratch2/nkudryas/BAND-torch/configs ~/experiments/BAND-torch/`

Even better though, prepare all code/configs in advance, push to a github branch, and pull them to the GPGPU slurm server. You'll need:
1. Model and datamodule configs
2. `scripts/run_<smth>.py`
3. updates `scripts/slurm/gen_experiment.py` to generate a new experiment.txt file for many parallel runs (e.g. multiple independent datasets or hyperparameter sets), if needed; or just 1 command to run a single script above (i.e. 1 line of experiment.txt). This line typically looks like:
`python scripts/run_pbt.py ${model} ${dataset} ${output_folder_name} ${T} ${fac_dim} ${co_dim} ${enc_dim} ${bw}`
But the run script can be customized to take more or less parameters.
It is a good idea to run LFADS & BAND in parallel, e.g. make the following experiment.txt file:
```
python scripts/run_pbt.py ${model} ${dataset} lfads_${output_folder_name} ${T} ${fac_dim} ${co_dim} ${enc_dim} 0.0
python scripts/run_pbt.py ${model} ${dataset} band_${output_folder_name} ${T} ${fac_dim} ${co_dim} ${enc_dim} 0.01
```
(with all the variables filled in with actual numerical values)
Then, the `run experiment` command will run both LFADS and BAND in parallel (see instructions below).

test if BAND runs:
  1. create an interactive session
  `srun --time=08:00:00 --mem=14000 --cpus-per-task=8 --gres=gpu:4 --pty bash`
  2. set up a micromamba environment (following the installation guidelines); if no internet connection on allocated server -- set up on the node. I set them up in `~/mamba_envs/` (on DFS).
  2. create `/disk/scratch/{USER}/BAND-torch`
  3. create `/disk/scratch/{USER}/BAND-torch/datasets` and copy relevant datasets there
  4. run `python scripts/run_pbt_slurm.py ....` with some relevant attribute (example in any line of `scripts/slurm/experiment.txt`)
  5. don't forget to copy runs back into `~/experiments/BAND-torch/runs/` before closing the session (if you train this until the end want to keep the result)

For parallel runs, use:
  1. `python scripts/slurm/gen_experiment.py` to generate a new experiment.txt file
  2. `run_experiment -b scripts/slurm/band_arrayjob.sh -e scripts/slurm/experiment.txt`

To use `run_experiment` don't forget to install clusterscripts before the first run, e.g.
```
echo 'export PATH=/home/$USER/git/cluster-scripts/experiments:$PATH' >> ~/.bashrc
source ~/.bashrc
```

Monitor the progress with:
1. `myjobs`
2. checking tail of the log files in `~/slurm_logs/`; try `tail -f ~/slurm_logs/<yourlog>` to watch it live.
more useful commands [here](https://github.com/cdt-data-science/cluster-scripts)

