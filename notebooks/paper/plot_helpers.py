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

def get_random_trials2plot(dir_index, epochs,epoch=1):
    '''
    Select random trials to plot

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
    for d in np.unique(dir_index):
        mask = (epochs == epoch) & (dir_index == d)
        ids_trials = np.where(mask)[0]
        if len(ids_trials) != 0:
            idx = np.random.choice(ids_trials)
            # print(idx)
            trials2plot[idx] = 1
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
    dataset_name=None,
    axes=None,
    title_y=-.33
):
    '''
    Plot the hand trajectories for 3 epochs (BL/AD/WO)
    '''
    dir_index = [
        sorted(set(ground_truth_directions)).index(i) for i in ground_truth_directions
    ]
    if axes is None:
        fig, axes = plt.subplots(1,3,figsize=(10, 3))

    for t in range(0,ground_truth_behaviours.shape[0]):
        axes[epochs[t]].plot(
                ground_truth_behaviours[t, :, 0],
                ground_truth_behaviours[t, :, 1],
                color=f"C{dir_index[t]}",
                alpha=1,
                ls = 'solid',
                linewidth=.5,
            )
        
    for i,ax in enumerate(axes):
        # make short axes arrows labeled x-y
        ax.axis("off")
        # get axis range
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        ax.quiver(xlim[0], ylim[0], (xlim[1]-xlim[0])/4, 0, angles="xy", scale_units="xy", scale=1)
        ax.quiver(xlim[0], ylim[0], 0, (ylim[1]-ylim[0])/4, angles="xy", scale_units="xy", scale=1)
        # label center of the arrows x and y (left and bottom)
        ax.text(xlim[0] + (xlim[1]-xlim[0])/8, ylim[0], "x", ha="center", va="top")
        ax.text(xlim[0], ylim[0] + (ylim[1]-ylim[0])/8, "y ", ha="right", va="center")

        if R2_pos is not None:
            ax.set_title(f'R2 = {100*R2_pos[i]:.2f}%')

    # add new axis on top of middle axis
    fig = ax.get_figure()
    p_center = axes[1].get_position()
    ax = fig.add_axes([p_center.x0, p_center.y0+p_center.height, 
                       p_center.width, p_center.height*0.1], frame_on=False)
    ax.axis("off")
    # add a bar for the central axes labelled perturbation
    ax.text(0.1, 0.01, "Perturbation", ha="center", va="bottom")
    ax.plot([-1,1], [0,0], color="black", lw=2)
    ax.set_xlim(-1,1)

    # label epochs with titles below subplots
    axes[0].set_title("Baseline (BL)",y=title_y)
    axes[1].set_title("Adaptation (AD)",y=title_y)
    axes[2].set_title("Washout (WO)",y=title_y)
    
    if dataset_name is not None:
        fig.savefig(f"{dataset_name}.pdf")
    # plt.close()


def plot_beh_pred_per_epoch(vel, pred_vel, dir_index, trials2plot, epochs, component=0, file_name=None,ax_vel=None):
    '''
    Plot hand velocity in the 3 epochs (BL/AD/WO)
    '''
    if ax_vel is None:
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

    if file_name:
        plt.savefig(file_name)

# FIGURE 4
    
