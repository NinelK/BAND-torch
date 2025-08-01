import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import fft, fftfreq

### SHARED FUNCTIONS


def get_trials2plot(pos, avg_pos, dir_index, epochs, epoch=1):
    """
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
    """
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


def get_random_trials2plot(dir_index, epochs, epoch=1):
    """
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
    """
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
    ground_truth_behaviours, ground_truth_directions, epoch, dataset_name=""
):
    """
    Plot the hand trajectories
    """
    dir_index = [
        sorted(set(ground_truth_directions)).index(i) for i in ground_truth_directions
    ]
    plt.figure(figsize=(5, 5))
    for t in range(0, ground_truth_behaviours.shape[0]):
        # ls = ':' if epoch[t]==0 else 'solid'
        if epoch[t]:
            plt.plot(
                ground_truth_behaviours[t, :, 0],
                ground_truth_behaviours[t, :, 1],
                color=f"C{dir_index[t]}",
                alpha=1,
                ls="solid",
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
    title_y=-0.33,
):
    """
    Plot the hand trajectories for 3 epochs (BL/AD/WO)
    """
    dir_index = [
        sorted(set(ground_truth_directions)).index(i) for i in ground_truth_directions
    ]
    if axes is None:
        fig, axes = plt.subplots(1, 3, figsize=(10, 3))

    for t in range(0, ground_truth_behaviours.shape[0]):
        axes[epochs[t]].plot(
            ground_truth_behaviours[t, :, 0],
            ground_truth_behaviours[t, :, 1],
            color=f"C{dir_index[t]}",
            alpha=1,
            ls="solid",
            linewidth=0.5,
        )

    for i, ax in enumerate(axes):
        # make short axes arrows labeled x-y
        ax.axis("off")
        # get axis range
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        ax.quiver(
            xlim[0],
            ylim[0],
            (xlim[1] - xlim[0]) / 4,
            0,
            angles="xy",
            scale_units="xy",
            scale=1,
        )
        ax.quiver(
            xlim[0],
            ylim[0],
            0,
            (ylim[1] - ylim[0]) / 4,
            angles="xy",
            scale_units="xy",
            scale=1,
        )
        # label center of the arrows x and y (left and bottom)
        ax.text(xlim[0] + (xlim[1] - xlim[0]) / 8, ylim[0], "x", ha="center", va="top")
        ax.text(
            xlim[0], ylim[0] + (ylim[1] - ylim[0]) / 8, "y ", ha="right", va="center"
        )

        if R2_pos is not None:
            ax.set_title(f"R2 = {100*R2_pos[i]:.2f}%")

    # add new axis on top of middle axis
    fig = ax.get_figure()
    p_center = axes[1].get_position()
    ax = fig.add_axes(
        [
            p_center.x0,
            p_center.y0 + p_center.height,
            p_center.width,
            p_center.height * 0.1,
        ],
        frame_on=False,
    )
    ax.axis("off")
    # add a bar for the central axes labelled perturbation
    ax.text(0.1, 0.01, "Perturbation", ha="center", va="bottom")
    ax.plot([-1, 1], [0, 0], color="black", lw=2)
    ax.set_xlim(-1, 1)

    # label epochs with titles below subplots
    axes[0].set_title("Baseline (BL)", y=title_y)
    axes[1].set_title("Adaptation (AD)", y=title_y)
    axes[2].set_title("Washout (WO)", y=title_y)

    if dataset_name is not None:
        fig.savefig(f"{dataset_name}.pdf")
    # plt.close()


def plot_beh_pred_per_epoch(
    vel,
    pred_vel,
    dir_index,
    trials2plot,
    epochs,
    component=0,
    file_name=None,
    ax_vel=None,
):
    """
    Plot hand velocity in the 3 epochs (BL/AD/WO)
    """

    from lfads_torch.metrics import r2_score

    if ax_vel is None:
        fig = plt.figure(figsize=(6, 3))

        ax_vel = [
            [fig.add_axes([0.00, 0.1 * i, 0.25, 0.1]) for i in range(8)],
            [fig.add_axes([0.33, 0.1 * i, 0.25, 0.1]) for i in range(8)],
            [fig.add_axes([0.67, 0.1 * i, 0.25, 0.1]) for i in range(8)],
        ]

    BIN_SIZE = 10  # ms
    time = np.arange(vel.shape[1]) * BIN_SIZE / 1000

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
        ax[0].spines["top"].set_visible(False)
        ax[0].spines["right"].set_visible(False)
        ax[0].set_yticks([])
        ax[0].set_xlabel("Time, s")
        ax[0].set_ylabel(f"Velocity {'X' if component == 0 else 'Y'}")

    R2s = []
    for e in range(3):
        mask = epochs == e
        R2_iso_vel = r2_score(pred_vel[mask], vel[mask])  # isotropic R2
        ax_vel[e][-1].set_title(r"$R^2_{vel}$ =" + f"{R2_iso_vel*100:.2f}%")
        R2s.append(R2_iso_vel)

    if file_name:
        plt.savefig(file_name)

    return R2s


# FIGURE 3


def plot_fourier_AD(axes, vel, epoch, experiments, dt=0.01):
    for ax, m, exps in zip(axes, ["C", "M"], [experiments[:4], experiments[4:]]):
        for i, spike_data_dir in enumerate(exps):
            # trial types are ordered in df
            AD_start = np.where(epoch[spike_data_dir] == "AD")[0][0]
            WO_start = np.where(epoch[spike_data_dir] == "WO")[0][0]

            V = vel[spike_data_dir][
                AD_start:WO_start, :
            ]  # get data for AD trials, time point 80 onwards
            T = dt * V.shape[1]

            SR = []
            for _ in range(100):
                idxs = np.random.choice(V.shape[0], V.shape[0], replace=True)
                Sxx_comp = []
                for c in [0, 1]:
                    x = V[idxs][..., c]

                    xf = fft(x)  # Compute Fourier transform of x
                    Sxx_all = (
                        2 * dt**2 / T * (xf * xf.conj()).real
                    )  # Compute power spectrum

                    Sxx_comp.append(Sxx_all)
                Sxx_comp = np.array(Sxx_comp)  # [components, trials, freqs]
                SR.append(np.sqrt(Sxx_comp.mean(0)).mean(0))

            faxis = fftfreq(Sxx_all.shape[1]) / dt  # Construct frequency axis
            SR = np.asarray(SR)

            mask = (faxis > 0) & (faxis <= 10)
            ax.plot(faxis[mask], SR.mean(0)[mask], c=f"C{i}")

            ax.fill_between(
                faxis[mask],
                SR.mean(0)[mask] - SR.std(0)[mask],
                SR.mean(0)[mask] + SR.std(0)[mask],
                alpha=0.3,
                color=f"C{i}",
                label=spike_data_dir.split("_")[3].split(".mat")[0][5:],
            )

        ax.set_xlabel("Frequency, Hz")
        ax.set_ylabel("FFT Amplitude, cm/s")
        ax.set_title(f"Monkey {m}")
        ax.set_ylim([0, 2.5])
        ax.set_xlim([0, 10])
        ax.legend()

    axes[0].arrow(5.0, 1.9, 0, -0.2, color="k", head_width=0.2, head_length=0.2)
    axes[1].arrow(4.0, 2, 0, -0.2, color="k", head_width=0.2, head_length=0.2)


def plot_fourier_last_sessions(
    ax,
    monkey,
    spike_data_dir,
    vel,
    epoch,
    dt=0.01,
    plot="spectrum",
    peak_freq=5,
    color=None,
    vmax=500,
):
    session = spike_data_dir.split("_")[3].split(".mat")[0][5:]
    epoch_args = {"fontsize": 12, "ha": "center"}

    # trial types are ordered in df
    AD_start = np.where(epoch[spike_data_dir] == "AD")[0][0]
    WO_start = np.where(epoch[spike_data_dir] == "WO")[0][0]

    V = vel[spike_data_dir][:, :]  # get data for all trials, time point 80 onwards
    T = dt * V.shape[1]
    # df = 1 / T  # Determine frequency resolution
    # fNQ = 1 / dt / 2  # Determine Nyquist frequency
    faxis = fftfreq(V.shape[1]) / dt  # Construct frequency axis
    find_osc = np.argmin(np.abs(faxis - peak_freq))

    SR = []
    Sxx_comp = []
    for c in [0, 1]:
        x = V[..., c]

        xf = fft(x)  # Compute Fourier transform of x
        Sxx_all = (xf * xf.conj()).real  # Compute spectrum

        Sxx_comp.append(Sxx_all)
    Sxx_comp = np.array(Sxx_comp)  # [components, trials, freqs]
    SR.append(np.sqrt(Sxx_comp.mean(0)).mean(0))

    faxis = fftfreq(Sxx_all.shape[1]) / dt  # Construct frequency axis
    SR = np.asarray(SR)

    SR = []
    for c in [0, 1]:
        x = V[..., c]
        xf = fft(x)  # Compute Fourier transform of x

        # print(np.sum((xf*xf.conj()).real)/np.sum(x**2)*dt/T) # check that power sums to velocity**2

        SR.append(np.sqrt((xf * xf.conj()).real))

    SR = np.asarray(SR)  # components, trials, frequencies

    mask = (faxis >= 0.0) & (faxis < 10.0)

    if plot == "spectrum":
        im = ax.imshow(
            SR.mean(0).T[mask], aspect="auto", cmap="plasma", vmin=0, vmax=vmax
        )
        plt.colorbar(
            im, ax=ax, extend="max", label="FFT amplitude", location="bottom", pad=0.25
        )
        ax.set_yticks(np.arange(0, len(faxis[mask]), 2))
        ax.set_yticklabels([f"{s:.2}" for s in faxis[mask][::2]])
        ax.set_ylabel("freq, Hz")
        # add arrow at peak frequency
        arrow_len, arrow_head = AD_start / 3, AD_start / 10
        ax.arrow(
            AD_start - 1.2 * arrow_len - arrow_head,
            find_osc,
            arrow_len,
            0,
            color="white",
            head_width=0.5,
            head_length=arrow_head,
        )
        # label epochs
        epoch_args["color"] = "white"
        epoch_args["va"] = "bottom"
        epoch_y = len(faxis[mask]) - 0.5

    elif plot == "top_freq":
        trials = np.arange(SR.shape[1])
        if ax is not None:
            ax.plot(trials, SR[..., find_osc].mean(0), c=color, alpha=0.5)
            # gaussian filter
            kernel = np.exp(-np.linspace(-3, 3, 50) ** 2)
            kernel /= kernel.sum()
            ax.plot(
                trials,
                np.convolve(SR[..., find_osc].mean(0), kernel, mode="same"),
                c=color,
                label="smoothed",
            )
            ax.set_ylabel(f"FFT Amplitude ({peak_freq} Hz)")
            epoch_args["color"] = "k"
            epoch_args["va"] = "top"
            epoch_y = ax.get_ylim()[1] * 0.95

        return trials, SR[..., find_osc].mean(0)

    ax.set_xlabel("trials, #")
    ax.set_title(f"Monkey {monkey} (session {session})")
    ax.axvline(AD_start, c="k")
    ax.axvline(WO_start, c="k")
    ax.text(AD_start / 2, epoch_y, "BL", **epoch_args)
    ax.text(AD_start + (WO_start - AD_start) / 2, epoch_y, "AD", **epoch_args)
    ax.text(WO_start + (SR.shape[1] - WO_start) / 2, epoch_y, "WO", **epoch_args)


# FIGURE 4


def class_accuracy(y_train, dir_index_train, y_pred, dir_index):
    """
    Train an LDA classifier to predict the target direction from the predicted velocities.
    """
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

    lda = LinearDiscriminantAnalysis()
    lda.fit(y_train, dir_index_train)
    return lda.score(y_pred, dir_index)


def plot_fourier_with_cos_sim(
    ax, ax_p, V, V_true, dt=0.01, label="", c="k", linestyle="solid", peak_freq=5
):
    #  text_pos = None):

    SR = []
    cos_sim = []
    for _ in range(100):
        idxs = np.random.choice(V.shape[0], V.shape[0], replace=True)

        x = V[idxs]
        xf = fft(x)  # Compute Fourier transform of x
        Sxx_all = (xf * xf.conj()).real  # Compute power spectrum

        SR.append(np.sqrt(Sxx_all).mean(0))

        if V_true is not None:
            x_true = V_true[idxs]
            xf_true = fft(x_true)  # Compute Fourier transform of x_true
            cos = np.cos(np.angle(xf) - np.angle(xf_true))
            cos_sim.append(cos.mean(0))

    faxis = fftfreq(Sxx_all.shape[1]) / dt  # Construct frequency axis
    SR = np.asarray(SR)  # [samples, freq]

    mask = (faxis > 0) & (faxis <= 10)
    ax.plot(faxis[mask], SR.mean(0)[mask], c=c, label=label, linestyle=linestyle)

    ax.fill_between(
        faxis[mask],
        SR.mean(0)[mask] - SR.std(0)[mask],
        SR.mean(0)[mask] + SR.std(0)[mask],
        alpha=0.3,
        color=c,
    )

    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel(r"FFT Amplitude [cm/s]")
    ax.set_ylim([0, 150])
    ax.set_xlim([0, 10])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True)

    if V_true is not None:
        cos_sim = np.asarray(cos_sim)  # [samples, freq]

        idx_peak = np.argmin(np.abs(faxis - peak_freq))
        str = label + " cos$_{" + f"{faxis[idx_peak]:.1f}" + "Hz}$="
        label_p = rf"{str}" + f"{cos_sim.mean(0)[idx_peak]:.2f}"

        ax_p.plot(
            faxis[mask], cos_sim.mean(0)[mask], c=c, label=label_p, linestyle=linestyle
        )

        ax_p.fill_between(
            faxis[mask],
            cos_sim.mean(0)[mask] - cos_sim.std(0)[mask],
            cos_sim.mean(0)[mask] + cos_sim.std(0)[mask],
            alpha=0.3,
            color=c,
        )
        ax_p.set_xlabel("Frequency [Hz]")
        ax_p.set_ylabel(r"FFT phase similarity")
        ax_p.set_ylim([-0.2, 1])
        ax_p.set_xlim([0, 10])
        ax_p.spines["top"].set_visible(False)
        ax_p.spines["right"].set_visible(False)
        ax_p.grid(True)
