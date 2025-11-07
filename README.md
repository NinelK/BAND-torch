# `BAND-torch`: Behavior-aligned neural dynamics model

The code in this repository extends [`lfads-torch`](https://github.com/arsedler9/lfads-torch) by Sedler et. al: A modular and extensible implementation of latent factor analysis via dynamical systems
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
`export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:<YOUR-PATH>/envs/band-torch/lib/`

To fix `ImportError: cannot import name 'packaging' from 'pkg_resources'`:
downgrade setuptools <70:
`micromamba install setuptools==69.5.1`

To fix `AttributeError: module 'numpy' has no attribute 'bool8'`:
downgrade numpy < 2
`pip install numpy==1.26.0 scikit-learn==1.3.0`
basically, `import sklearn` won't work without these downgrades. Until sklearn can't be imported, BAND will throw confusing hydra-related errors.

Backward compatibility for loading older models with newer PyTorch:
`export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`


Monitor the progress with:
1. `myjobs`
2. checking tail of the log files in `~/slurm_logs/`; try `tail -f ~/slurm_logs/<yourlog>` to watch it live.
more useful commands [here](https://github.com/cdt-data-science/cluster-scripts)

