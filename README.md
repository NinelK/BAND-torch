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

`python ./scripts/lorenz/generate_lorenz.py --delay_bins=-10 --off_manifold_kick_magnitude=5.`

`python ./scripts/lorenz/generate_lorenz.py --delay_bins=0 --off_manifold_kick_magnitude=5.`

`python ./scripts/lorenz/generate_lorenz.py --delay_bins=10 --off_manifold_kick_magnitude=5.`

2. Run CSAE and BAND:
`sh ./scripts/tune_synthetic_datasets.sh`

3. Analyse the results:

# Mapping notebooks to figures

## Main text figures
1. Vector neural code schematics, no code
2. `notebooks/paper/Fig2_vel_oscillations.ipynb`
3. `notebooks/paper/Fig3_decoding.ipynb`
4. Vector model schematics, no code
5. `notebooks/paper/Fig5_supervision.ipynb`
6. `notebooks/paper/Fig6_small_controlled_variability.ipynb`
7. `notebooks/paper/Fig7_lags.ipynb`

## Supplemental figures
1. `notebooks/paper/SFig1_to_6_decoding_across_animals.ipynb`
2. `notebooks/paper/SFig1_to_6_decoding_across_animals.ipynb`
3. `notebooks/paper/SFig1_to_6_decoding_across_animals.ipynb`
4. `notebooks/rnn_decoder/RNN_decoder.ipynb`
5. `notebooks/paper/SFig1_to_6_decoding_across_animals.ipynb`
6. `notebooks/paper/SFig1_to_6_decoding_across_animals.ipynb`
7. `notebooks/paper/SFig7_no_control.ipynb`
8. `notebooks/paper/Fig6_small_controlled_variability.ipynb`
9. `notebooks/paper/SFig9_across_animals.ipynb`
10. `notebooks/paper/Fig5_supervision.ipynb`
11. `notebooks/paper/Fig6_small_controlled_variability.ipynb`
12. `notebooks/paper/Fig7_lags.ipynb`
13. `notebooks/paper/SFig13_residuals.ipynb`
14. `notebooks/paper/Fig3_decoding.ipynb`
15. `notebooks/paper/SFig15_to_16_synthetic_data_lorenz.ipynb`
16. `notebooks/paper/SFig15_to_16_synthetic_data_lorenz.ipynb`
17. Adapted from the official challange leaderboard: https://neurallatents.github.io/; https://eval.ai/web/challenges/challenge-page/1256/
18. Vector model schematics, no code

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
