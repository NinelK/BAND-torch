import math

import torch
from torch.nn.functional import poisson_nll_loss
from torchmetrics import Metric

import numpy as np


def r2_score(preds, targets):
    '''
    Computes an isotropic R2 metric
    (almost like a classic anisotropic one, but isotropic)
    '''
    if preds.ndim > 2:
        preds = preds.reshape(-1, preds.shape[-1])
    if targets.ndim > 2:
        targets = targets.reshape(-1, targets.shape[-1])
    if type(preds) == torch.Tensor:
        target_mean = torch.mean(targets, dim=0)
        ss_tot = torch.sum((targets - target_mean) ** 2, dim=0)
        ss_res = torch.sum((targets - preds) ** 2, dim=0)
        return torch.mean(1 - ss_res / ss_tot)
    elif type(preds) == np.ndarray:
        target_mean = np.mean(targets, axis=0)
        ss_tot = np.sum((targets - target_mean) ** 2, axis=0)
        ss_res = np.sum((targets - preds) ** 2, axis=0)
        return np.mean(1 - ss_res / ss_tot)

def r2_UIVE(preds, targets, dir_index):
    '''
    UnInstructed Variance Explained for reaches to 8 directions
    '''
    assert np.allclose(np.arange(8),np.unique(dir_index))
    avg_vel = [np.mean(targets[dir_index==d], axis=0) for d in range(8)]
    total_var = [np.sum((targets[dir_index==d] - avg_vel[d])**2) for d in range(8)]
    expl_var  = [np.sum((preds[dir_index==d] - targets[dir_index==d])**2) for d in range(8)] 
    if np.allclose(total_var,0) & np.allclose(expl_var,0):
        R2_UIVE = 0
    else:
        for d in range(8):
            if total_var[d] == 0:
                total_var[d] = np.nan
        r2_uive = [1 - expl_var[d] / total_var[d] for d in range(8)]
        R2_UIVE = np.nanmean(r2_uive)
        if np.any(r2_uive) == np.nan:
            print('Warning: Some directions have no variance in the data: ',np.isnan(np.array(r2_uive)))
    return R2_UIVE

def bits_per_spike(preds, targets):
    """
    Computes BPS for n_samples x n_timesteps x n_neurons arrays.
    Preds are logrates and targets are binned spike counts.
    """
    nll_model = poisson_nll_loss(preds, targets, full=True, reduction="sum")
    nll_null = poisson_nll_loss(
        torch.mean(targets, dim=(0, 1), keepdim=True),
        targets,
        log_input=False,
        full=True,
        reduction="sum",
    )
    return (nll_null - nll_model) / torch.nansum(targets) / math.log(2)


def regional_bits_per_spike(preds, targets, encod_data_dim, encod_seq_len):
    """
    Computes BPS for n_samples x n_timesteps x n_neurons arrays, separated into
    heldin and heldout regions.
    """
    edd, esl = encod_data_dim, encod_seq_len
    bps = bits_per_spike(preds[:, :esl, :edd], targets[:, :esl, :edd])
    co_bps = bits_per_spike(preds[:, :esl, edd:], targets[:, :esl, edd:])
    fp_bps = bits_per_spike(preds[:, esl:], targets[:, esl:])
    return bps, co_bps, fp_bps


class ExpSmoothedMetric(Metric):
    """Averages within epochs and exponentially smooths between epochs."""

    def __init__(self, coef=0.9, **kwargs):
        super().__init__(**kwargs)
        self.coef = coef
        # PTL will automatically `reset` these after each epoch
        self.add_state("value", default=torch.tensor(0.0))
        self.add_state("count", default=torch.tensor(0))
        # Previous value must be immune to `reset`
        self.prev = torch.tensor(float("nan"))

    def update(self, value, batch_size):
        self.value += value * batch_size
        self.count += batch_size

    def compute(self):
        curr = self.value / self.count
        if torch.isnan(self.prev):
            self.prev = curr
        smth = self.coef * self.prev + (1 - self.coef) * curr
        self.prev = smth
        return smth
