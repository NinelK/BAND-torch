import numpy as np
import matplotlib.pyplot as plt

### SHARED FUNCTIONS

def get_trials2plot(pos, avg_pos, dir_index, epochs,epoch=1):
    '''
    Select the trials to plot based on the distance between 
    the average position and the position in single trials

    Parameters
    ----------
    pos : np.ndarray
        The position in single trials
    avg_pos : np.ndarray
        The average position
    dir_index : np.ndarray
        The direction index
    epochs : np.ndarray
        The epochs
    epoch : int, optional
        The epoch to consider, by default 1 (adaptation)

    Returns
    -------
    np.ndarray
        The trials to plot
    '''
    trials2plot = np.zeros_like(epochs)
    for e in np.unique(epochs):
        for d in np.unique(dir_index):
            mask = (epochs == epoch) & (dir_index == d)
            # print(mask)
            dist = ((pos - avg_pos) ** 2).sum(-1).sum(-1)
            dist[~mask] = -np.inf
            # print(dist)
            idx_max = np.argmax(dist)
            # print(idx_max)
            trials2plot[idx_max] = 1
    return trials2plot

def plot_directions(
    ground_truth_behaviours,
    ground_truth_directions,
    epoch,
    dataset_name=''
):
    '''
    Plot the hand trajectories
    '''
    dir_index = [
        sorted(set(ground_truth_directions)).index(i) for i in ground_truth_directions
    ]
    plt.figure(figsize=(5, 5))
    for t in range(0,ground_truth_behaviours.shape[0]):
        # ls = ':' if epoch[t]==0 else 'solid'
        if epoch[t]:
            plt.plot(
                ground_truth_behaviours[t, :, 0],
                ground_truth_behaviours[t, :, 1],
                color=f"C{dir_index[t]}",
                alpha=1,
                ls = 'solid'
            )
    plt.axis("off")
    
    plt.savefig(f"{dataset_name}.pdf")

### FIGURE 3

def plot_directions_per_epoch(
    ground_truth_behaviours,
    ground_truth_directions,
    epochs,
    R2_pos=None,
    dataset_name=''
):
    '''
    Plot the hand trajectories for 3 epochs (BL/AD/WO)
    '''
    dir_index = [
        sorted(set(ground_truth_directions)).index(i) for i in ground_truth_directions
    ]
    fig, axes = plt.subplots(1,3,figsize=(10, 3))

    for t in range(0,ground_truth_behaviours.shape[0]):
        axes[epochs[t]].plot(
                ground_truth_behaviours[t, :, 0],
                ground_truth_behaviours[t, :, 1],
                color=f"C{dir_index[t]}",
                alpha=1,
                ls = 'solid',
                linewidth=1,
            )
    for i,ax in enumerate(axes):
        ax.axis("off")
        if R2_pos is not None:
            ax.set_title(f'R2 = {100*R2_pos[i]:.2f}%')
    
    fig.savefig(f"{dataset_name}.pdf")
    # plt.close()


def plot_beh_pred_per_epoch(vel, pred_vel, dir_index, trials2plot, epochs, component=0, file_name=""):
    '''
    Plot hand velocity in the 3 epochs (BL/AD/WO)
    '''
    fig = plt.figure(figsize=(6, 3))

    ax_vel = [
        [fig.add_axes([0.00, 0.1 * i, 0.25, 0.1]) for i in range(8)],
        [fig.add_axes([0.33, 0.1 * i, 0.25, 0.1]) for i in range(8)],
        [fig.add_axes([0.67, 0.1 * i, 0.25, 0.1]) for i in range(8)],
    ]

    BIN_SIZE = 10 # ms
    time = np.arange(vel.shape[1]) * BIN_SIZE

    for v, ls in zip([vel, pred_vel], ["--", "solid"]):
        for t in range(0, vel.shape[0]):
            # ls = ':' if epoch[t]==0 else 'solid'
            if trials2plot[t]:
                d = dir_index[t]
                ax_vel[epochs[t]][d].plot(
                    time,
                    v[t, :, component],
                    color=f"C{d}",
                    alpha=1,
                    ls=ls,
                )

    for ax in ax_vel:
        for a in ax[1:]:
            a.axis("off")
        ax[0].spines['top'].set_visible(False)
        ax[0].spines['right'].set_visible(False)
        ax[0].set_yticks([])
        ax[0].set_xlabel("Time (ms)")
        ax[0].set_ylabel("Velocity X")

    for e in range(3):
        mask = epochs == e
        R2_iso_vel = 1 - np.sum((vel[mask] - pred_vel[mask]) ** 2) / np.sum((vel[mask] - vel[mask].mean()) ** 2)
        ax_vel[e][-1].set_title(f'R2_vel = {R2_iso_vel*100:.2f}%')

    plt.savefig(file_name)

# FIGURE 4
    
