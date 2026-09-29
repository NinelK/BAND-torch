# compute PSTHs
import numpy as np
import torch


def input_norm_around_stim(controls, kick_idxs, t_before=10, t_after=20):
    co_dim = controls.shape[2]
    PSTH = np.zeros((co_dim, t_before + t_after))
    for t in range(len(kick_idxs)):
        for l in kick_idxs[t]:
            if l - t_before >= 0 and l + t_after < controls.shape[1]:
                PSTH += controls[t, l - t_before : l + t_after, :].T
    PSTH /= float(len(kick_idxs))
    return np.linalg.norm((PSTH.T - PSTH.T[:t_before].mean(0)), axis=1)


def r2_score(preds, targets):
    """
    Computes an isotropic R2 metric
    (almost like a classic anisotropic one, but isotropic)
    """
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


from scipy.stats import poisson


def bps(counts, rates):
    valid_mean_count = counts.mean(0).mean(0)  # to get Hz -> x100
    base_LL = poisson.logpmf(counts, valid_mean_count)
    n_sp = counts.sum()

    valid_LL = poisson.logpmf(counts, rates)
    valid_co_bps = np.nansum(valid_LL - base_LL) / (n_sp * np.log(2))

    return valid_co_bps
