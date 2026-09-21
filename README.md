# `BAND-torch`: Behavior-aligned neural dynamics model

BAND is a latent dynamics model weakly supervised with behavior. Using a well-established latent dynamics model (LFADS) as a baseline, we constructed an architecture that not only explains neural variability but also aligns the latent space with behavioral output (see the BAND schema for the new additions in green).

While standard LFADS can infer inputs through its controller RNN, it typically only captures inputs that cause a significant change in future dynamics and directly affect neural reconstruction. To ensure that behavior-related inputs are reliably captured, even if they cause only small, transient changes in neural dynamics, BAND incorporates an additional behavior decoder (linear or seq2seq).

The code in this repository extends [`lfads-torch`](https://github.com/arsedler9/lfads-torch) by Sedler et. al: A modular and extensible implementation of latent factor analysis via dynamical systems
[![arXiv](https://img.shields.io/badge/arXiv-2309.01230-b31b1b.svg)](https://arxiv.org/abs/2309.01230)

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

# Reproducing paper figures

The notebooks producing paper figures can be found in:
`/notebooks/paper/`
The names correspond to the figures in the manuscript.

## Downloading the Dataset
The preprocessed neural and behavioral data is hosted as a GitHub Release in this repository.

### Option 1: Command Line (Recommended)
You can download and extract the dataset directly into your project directory using wget and unzip:
```
# Download the dataset from the release assets
wget https://github.com/NinelK/BAND-torch/releases/download/dataset/CM.zip

# Extract the contents
unzip CM.zip -d datasets/

# Optional: Remove the zip file to save space
rm CM.zip
```

### Option 2: Manual Download

1. Navigate to the Releases page of this repository.
2. Find the release tagged dataset (Force field sessions).
3. Under the Assets section at the bottom of the release notes, click on CM.zip to download it.
4. Extract the downloaded .zip file into your local workspace.

For details on the file naming convention and how to load the .h5 files in Python, see the [release notes](https://github.com/NinelK/BAND-torch/releases/tag/dataset).

# Run synthetic examples
1. Generate spike data from a system with a latent Lorenz attractor:
`python ./scripts/lorenz/generate_lorenz.py --delay_bins=-10`
`python ./scripts/lorenz/generate_lorenz.py --delay_bins=0`
`python ./scripts/lorenz/generate_lorenz.py --delay_bins=10`
2. Run CSAE and BAND:
`sh ./scripts/tune_synthetic_datasets.sh`
3. Analyse the results:
`./notebooks/post/compare_pairs.ipynb`

# Notes on fixing problems

To fix `/lib64/libstdc++.so.6: version `CXXABI_1.3.9'` error, add path to this library in your env, e.g.:
`export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:<YOUR-PATH>/envs/band-torch/lib/`

To fix `ImportError: cannot import name 'packaging' from 'pkg_resources'`:
downgrade setuptools <70:
`micromamba install setuptools==69.5.1`

To fix `AttributeError: module 'numpy' has no attribute 'bool8'`:
downgrade numpy < 2
`pip install numpy==1.26.0 scikit-learn==1.3.0`
basically, `import sklearn` won't work without these downgrades. Until sklearn can't be imported, BAND will throw confusing hydra-related errors.

Backward compatibility for newer PyTorch:
`export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`
