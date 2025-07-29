import numpy as np
import pyaldata
from numpy.fft import fft, fftfreq

# just going to hard code this obviously change this if the data changes
experiments = [
    "Chewie_CO_FF_2016-09-15.mat",
    "Chewie_CO_FF_2016-09-21.mat",
    "Chewie_CO_FF_2016-10-05.mat",
    "Chewie_CO_FF_2016-10-07.mat",  # the best!
    "Mihili_CO_FF_2014-02-03.mat",  # *
    "Mihili_CO_FF_2014-02-17.mat",  # + (only BL)
    "Mihili_CO_FF_2014-02-18.mat",
    "Mihili_CO_FF_2014-03-07.mat",
]

raw_data_dir = "/disk/data1/motor_cortex/perich_2018_binned/"
data_save_dir = "./datasets/"


def spike_2_fr(spike_data, dt=0.01):
    FR = np.zeros_like(spike_data, dtype="float")
    kernel = np.exp(-np.linspace(-3, 3, 18) ** 2)
    kernel /= np.sum(kernel)
    for t in range(spike_data.shape[0]):
        for n in range(spike_data.shape[2]):
            spikes = spike_data[t, :, n]
            FR[t, :, n] = np.convolve(spikes, kernel, mode="same") * (1 / dt)
    return FR


for spike_data_dir in experiments[:]:
    results = {}

    """
    activity_key : which neurons to load {"PMd_spikes", "M1_spikes", "all_spikes"}
    behaviour_key : behaviour to train to (sometimes assumed to be velocity)
    full_behaviour_key : the other behaviour (vel or pos). Here often assumed to be position
    alignment : align trials to this point in the experinment. {"go"}
    """
    activity_key = "all_spikes"  # "PMd_spikes" # "all_spikes" #"M1_spikes"
    behaviour_key = "vel"  # the main behavior used in training & validation
    full_behaviour_key = "pos"  # extra variable used for validation

    """
    perc_train : percentage of trials to use for training
    perc_valid/test : percentage of trials to use for valid/test sets. In some code if perc_test==0 then test set will be equivalent to valid set (stops lfads complaining)
    """
    perc_train = 80
    perc_valid = 20
    perc_test = 0

    """
    start_key : point to align trials to
    shortest_trial : length of trials once aligned (in bins, 10ms assumed)
    before_align_point : time to keep before alignment point (in bins, 10ms assumed)
    """
    start_key = "idx_go_cue"  #'idx_peak_speed' #
    before_align_point = 20

    """
    dataset_type : keep all epochs? session is all epochs, can also do BLAD (only baseline and adaptation trials)
    """
    dataset_type = "session"

    longest_trial = None  # dont know

    ablation_study = True
    oscillation_threshold = (
        0.6  # doesnt matter (except plotting) unless ablation_study==True
    )

    """
    dt : time step size - this actually depends on the bin window but I cant find where this is set - it is 10ms rn
    """
    dt = 0.01

    """
    n_shuffles : number of shuffles to do
    """
    n_shuffles = 100

    # Fourier

    time_window_start = 80  # this is the time point to start the window for the FFT (())d[start_key] - before_align_point : d[start_key] + shortest_trial

    # load data
    pd_data = pyaldata.io.mat2dataframe(
        path=raw_data_dir + spike_data_dir, shift_idx_fields=True
    )
    if activity_key == "all_spikes":
        pd_data[activity_key] = [
            np.concatenate([m1, pmd], axis=1)
            for m1, pmd in zip(pd_data["M1_spikes"], pd_data["PMd_spikes"])
        ]
        print(
            pd_data["M1_spikes"][0].shape,
            pd_data["PMd_spikes"][0].shape,
            pd_data["all_spikes"][0].shape,
        )

    num_m1_units = pd_data["M1_spikes"][0].shape[1]

    # sanity check train/valid/test settings
    assert perc_train + perc_valid + perc_test == 100

    # needed for plots, must align with chosen region
    activity_key_to_area = {"PMd_spikes": "PMd", "M1_spikes": "M1", "all_spikes": "all"}
    area = activity_key_to_area[activity_key]  # 'M1' #'PMd' # 'all' # 'M1'

    # trials must be successful to be included
    selected_trials = pd_data.loc[(pd_data.result == "R")]  # & (pd_data.epoch == "AD")

    shortest_trial = int(
        (selected_trials.idx_trial_end - selected_trials[start_key]).min()
    )
    print(start_key, shortest_trial)

    # spike data from area, aligned to start key such that with window [- before_align_point : start_key : + shortest trial ]
    spike_data = np.asarray(
        [
            d[f"{area}_spikes"][
                d[start_key] - before_align_point : d[start_key] + shortest_trial, :
            ]
            for i, d in selected_trials.iterrows()
        ]
    )
    print(spike_data.shape)
    # time between start key and go cue
    target_on = np.asarray(
        [d[start_key] - d["idx_go_cue"] for i, d in selected_trials.iterrows()]
    )
    # epoch (BL/AD/WO) for each trial
    epoch = np.asarray([d["epoch"] for i, d in selected_trials.iterrows()])
    # trial types are ordered in df
    AD_start = np.where(epoch == "AD")[0][0]
    WO_start = np.where(epoch == "WO")[0][0]

    # velocity of hand, aligned to start key as above
    vel = np.asarray(
        [
            d["vel"][d[start_key] - before_align_point : d[start_key] + shortest_trial]
            for i, d in selected_trials.iterrows()
        ]
    )

    # SHUFFLE spikes
    power4_shuffled, power5_shuffled = [], []
    power4_BL_shuffled, power5_BL_shuffled = [], []
    for _ in range(n_shuffles):
        shuffled_spikes = np.zeros_like(spike_data)
        for t in range(spike_data.shape[0]):
            for n in range(spike_data.shape[2]):
                shuffled_spikes[t, :, n] = np.random.permutation(spike_data[t, :, n])

        fr = spike_2_fr(shuffled_spikes[:AD_start])  # get data for BL trials
        T = dt * fr.shape[1]

        power4_BL, power5_BL = [], []
        for i in range(fr.shape[-1]):  # for each neuron
            x = fr[:, :, i]  # get the AD spike data for neuron i

            xf = fft(x)  # Compute Fourier transform of x
            Sxx = (xf * xf.conj()).real  # Compute spectrum
            Sxx = np.nanmean(Sxx, axis=0)  # trial average per epoch

            df = 1 / T  # Determine frequency resolution
            faxis = fftfreq(len(Sxx)) / dt  # Construct frequency axis

            j5 = np.argmin(np.abs(faxis - 5))
            power5_BL.append(Sxx.real[j5])  # power at 5Hz
            j4 = np.argmin(np.abs(faxis - 4))
            power4_BL.append(Sxx.real[j4])

        power5_BL_shuffled.append(power5_BL)
        power4_BL_shuffled.append(power4_BL)

        fr = spike_2_fr(shuffled_spikes[AD_start:WO_start])  # get data for AD trials
        T = dt * fr.shape[1]

        power4, power5 = [], []
        for i in range(fr.shape[-1]):  # for each neuron
            x = fr[:, :, i]  # get the AD spike data for neuron i

            xf = fft(x)  # Compute Fourier transform of x
            Sxx = (xf * xf.conj()).real  # Compute spectrum
            Sxx = np.nanmean(Sxx, axis=0)  # trial average per epoch

            df = 1 / T  # Determine frequency resolution
            faxis = fftfreq(len(Sxx)) / dt  # Construct frequency axis

            j5 = np.argmin(np.abs(faxis - 5))
            power5.append(Sxx.real[j5])  # power at 5Hz
            j4 = np.argmin(np.abs(faxis - 4))
            power4.append(Sxx.real[j4])

        power5_shuffled.append(power5)
        power4_shuffled.append(power4)

    power5_shuffled = np.asarray(power5_shuffled)
    power4_shuffled = np.asarray(power4_shuffled)

    power5_BL_shuffled = np.asarray(power5_BL_shuffled)
    power4_BL_shuffled = np.asarray(power4_BL_shuffled)

    # Find 5Hz mode
    fr_BL = spike_2_fr(spike_data[:AD_start])  # get data for BL trials
    fr = spike_2_fr(spike_data[AD_start:WO_start])  # get data for AD trials
    T = dt * fr.shape[1]

    results = []
    results_weak = []
    oscillating_score = np.zeros((2, 2, fr.shape[-1]))
    for i in range(fr.shape[-1]):  # for each neuron
        x = fr_BL[:, :, i]  # get the AD spike data for neuron i

        xf = fft(x)  # Compute Fourier transform of x
        Sxx_all = (xf * xf.conj()).real  # Compute spectrum

        Sxx = np.nanmean(Sxx_all, axis=0)  # trial average per epoch

        df = 1 / T  # Determine frequency resolution
        faxis = fftfreq(len(Sxx)) / dt  # Construct frequency axis

        j5 = np.argmin(np.abs(faxis - 5))
        power5_BL = Sxx[j5]
        j4 = np.argmin(np.abs(faxis - 4))
        power4_BL = Sxx[j4]

        x = fr[:, :, i]  # get the AD spike data for neuron i

        xf = fft(x)  # Compute Fourier transform of x
        Sxx_all = (xf * xf.conj()).real  # Compute spectrum

        Sxx = np.nanmean(Sxx_all, axis=0)  # trial average per epoch

        df = 1 / T  # Determine frequency resolution
        faxis = fftfreq(len(Sxx)) / dt  # Construct frequency axis

        j5 = np.argmin(np.abs(faxis - 5))
        power5 = Sxx[j5]
        j4 = np.argmin(np.abs(faxis - 4))
        power4 = Sxx[j4]

        thr5_BL = power5_BL_shuffled[:, i].mean() + 4 * power5_BL_shuffled[:, i].std()
        thr5 = power5_shuffled[:, i].mean() + 4 * power5_shuffled[:, i].std()
        thr4 = power4_shuffled[:, i].mean() + 4 * power4_shuffled[:, i].std()
        thr4_BL = power4_BL_shuffled[:, i].mean() + 4 * power4_BL_shuffled[:, i].std()
        if (power4 >= thr4) & (power4_BL < thr4_BL):
            results.append((faxis[j4], i))
        if (power5 >= thr5) & (power5_BL < thr5_BL):
            results.append((faxis[j5], i))
        thr5_BL = power5_BL_shuffled[:, i].mean() + 2 * power5_BL_shuffled[:, i].std()
        thr5 = power5_shuffled[:, i].mean() + 2 * power5_shuffled[:, i].std()
        thr4 = power4_shuffled[:, i].mean() + 2 * power4_shuffled[:, i].std()
        thr4_BL = power4_BL_shuffled[:, i].mean() + 2 * power4_BL_shuffled[:, i].std()
        if (power4 >= thr4) & (power4_BL < thr4_BL):
            results_weak.append((faxis[j4], i))
        if (power5 >= thr5) & (power5_BL < thr5_BL):
            results_weak.append((faxis[j5], i))

        oscillating_score[0, 0, i] = (
            power5 - power5_shuffled[:, i].mean()
        ) / power5_shuffled[:, i].std()
        oscillating_score[0, 1, i] = (
            power4 - power4_shuffled[:, i].mean()
        ) / power4_shuffled[:, i].std()
        oscillating_score[1, 0, i] = (
            power5_BL - power5_BL_shuffled[:, i].mean()
        ) / power5_BL_shuffled[:, i].std()
        oscillating_score[1, 1, i] = (
            power4_BL - power4_BL_shuffled[:, i].mean()
        ) / power4_BL_shuffled[:, i].std()

    results = np.array(results)
    results_weak = np.array(results_weak)
    # save summary
    with open(f"./results/spiking_{spike_data_dir.split('.mat')[0]}.csv", "w") as f:
        m1 = pd_data["M1_spikes"][0].shape[1]
        f.write(f"M1 neurons: {m1}\n")
        if len(results) > 0:
            for r in results:
                f.write(f"{r[0]:.1f} Hz: {int(r[1]):d},\t")
            f.write("\n")
            U = np.unique(results[..., 1])
            f.write(f"{np.sum(U<=m1)},\t{np.sum(U>m1)}")
            f.write("\n")
            for r in results_weak:
                f.write(f"{r[0]:.1f} Hz: {int(r[1]):d},\t")
            f.write("\n")
            U = np.unique(results_weak[..., 1])
            f.write(f"{np.sum(U<=m1)},\t{np.sum(U>m1)}")

    np.save(
        f"./results/oscillating_score_{spike_data_dir.split('.mat')[0]}.npy",
        oscillating_score,
    )
