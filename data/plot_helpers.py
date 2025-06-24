import matplotlib.pyplot as plt
from matplotlib.axes import Axes

import numpy as np
from lfads_torch.metrics import r2_score

def get_target_ids(true_target_direction):
    ''' substitute direction elements with ids '''
    uniq_dirs = np.unique(true_target_direction)
    true_label = np.array([np.where(uniq_dirs==t)[0][0] for t in true_target_direction])
    return true_label

def plot_avg_traj(data,true_target_direction,title='',epoch_mask=True, sharey=False):
    ''' plot average trajectory for each target direction '''
    true_label = get_target_ids(true_target_direction)
    n = data.shape[-1]
    col, row = min(n,5), max(1,int(np.ceil(n/5)))
    fig, ax = plt.subplots(row,col, figsize=(col*3,row*2),sharex=True, sharey=sharey)
    if type(ax) is Axes:
        ax = [ax]
    else:
        ax = ax.flatten()
    assert len(ax)>=n, f'not enough axes created for {n} factors'
    for i in range(n):
        for j in np.unique(true_label):
            ax[i].plot(data[(true_label==j) & epoch_mask].mean(0)[...,i],label=f'{true_target_direction[true_label==j][0]:.2f}')
        ax[i].set_title(f'{title} {i}')
        if i==(col-1):
            ax[i].legend(loc = (1.01,0))
    for i in range(n,len(ax)):
        ax[i].axis("off")
    fig.tight_layout()
    return fig

def get_trials2plot(pos, pred_pos, dir_index):
    trials2plot = np.zeros(pos.shape[0])
    for d in np.unique(dir_index):
        mask = (dir_index == d)
        # print(mask)
        dist = ((pos - pred_pos) ** 2).sum(-1).sum(-1)
        dist[~mask] = -np.inf
        # print(dist)
        idx_max = np.argmax(dist)
        # print(idx_max)
        trials2plot[idx_max] = 1
    return trials2plot

def plot_beh_pred(vel, pred_vel, dir_index, trials2plot, file_name=""):
    pos = np.cumsum(vel * 0.01, 1)
    pred_pos = np.cumsum(pred_vel * 0.01, 1)

    fig = plt.figure(figsize=(6, 3))

    axes = [
        fig.add_axes([0, 0, 0.5, 1])
    ]

    ax_vel = [
        [fig.add_axes([0.50, 0.1 * i, 0.25, 0.1]) for i in range(dir_index.max()+1)],
        [fig.add_axes([0.75, 0.1 * i, 0.25, 0.1]) for i in range(dir_index.max()+1)],
    ]

    time = np.arange(pos.shape[1]) * 10

    for p, v, ls in zip([pos, pred_pos], [vel, pred_vel], [":", "solid"]):
        for t in range(0, pos.shape[0]):
            # ls = ':' if epoch[t]==0 else 'solid'
            if trials2plot[t]:
                axes[0].plot(
                    p[t, :, 0],
                    p[t, :, 1],
                    color=f"C{dir_index[t]}",
                    alpha=1,
                    ls=ls,
                )
                d = dir_index[t]
                for i in range(2):
                    ax_vel[i][d].plot(
                        time,
                        v[t, :, i],
                        color=f"C{d}",
                        alpha=1,
                        ls=ls,
                    )

    for ax in axes:
        ax.axis("off")

    for ax in ax_vel:
        for a in ax:
            a.axis("off")

    R2_iso_pos = r2_score(pred_pos, pos)
    R2_iso_vel = r2_score(pred_vel, vel)
    
    axes[0].text(np.min(pos[...,0]),np.max(pos[...,1]),f'R2_pos = {R2_iso_pos*100:.2f}%')
    ax_vel[0][-1].set_title(f'R2_vel = {R2_iso_vel*100:.2f}%')

    plt.savefig(file_name)

