import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import h5py
from rnn_decoder import *

## Setups ################################
experiments = [
    'Chewie_CO_FF_2016-09-15.mat',
    'Chewie_CO_FF_2016-09-21.mat',
    'Chewie_CO_FF_2016-10-05.mat',
    'Chewie_CO_FF_2016-10-07.mat', # *
    'Chewie_CO_VR_2016-09-09.mat',
    'Chewie_CO_VR_2016-09-12.mat', 
    'Chewie_CO_VR_2016-09-14.mat',
    'Chewie_CO_VR_2016-10-06.mat',
    'Mihili_CO_FF_2014-02-03.mat', # *
    'Mihili_CO_FF_2014-02-17.mat', # + (only BL)
    'Mihili_CO_FF_2014-02-18.mat',
    'Mihili_CO_FF_2014-03-07.mat',
    'Mihili_CO_VR_2014-03-03.mat',
    'Mihili_CO_VR_2014-03-04.mat',
    'Mihili_CO_VR_2014-03-06.mat'
]
# dataset_name = "./data/Chewie_CO_FF_2016-10-05_BLAD_vel_pos_all_spikes_go/neural_BLAD"
# dataset_name = "./data/Jenkins/neural_all"
load_model = False
train = True
plotting = True
beh = ""

####################################
# for experiment in experiments:
# dataset_name = f'./data/{experiment[:-4]}_BLAD_vel_pos_all_spikes_go/neural_BLAD'

dataset_name = f'./data/Chewie_CO_FF_2016-10-07_BLAD_vel_pos_all_spikes_go/neural_BLAD'

# label, run = '_lfads_hist0_f32', '11-16-2022-12:48:22-run-hist0-32-0-rep-20'
label, run = '_tncdm_diag_hist0_f32', '11-16-2022-15:01:06-run-hist0-32-0.001-rep-23'
# label, run = '_tncdm_full_hist0_f32', '11-15-2022-23:55:51-run-hist0-32-0.001-rep-1'

rate_dir = '../ctrl_tndm/exp_results_Chewie_CO_FF_2016-10-07_BLAD_vel_pos_all_spikes_go/neural_BLAD.h5/ctrl_tndm_single/'+run

data = h5py.File(f"{dataset_name}.h5", "r")
# train_data = data["train_data"][:]  # (num_trials, len_trial, num_neurons)
# valid_data = data["valid_data"][:]
# test_data = data["test_data"][:]

train_target = data["train_target_direction"][:]
valid_target = data["valid_target_direction"][:]
test_target = data["test_target_direction"][:]

train_behaviours = data[f"train{beh}_behavior"][:]
valid_behaviours = data[f"valid{beh}_behavior"][:]
test_behaviours = data[f"test{beh}_behavior"][:]

mode = 'train'
key = 'factors'
train_data_h5 = h5py.File(f'{rate_dir}/model_runs__train_{mode}', 'r')
train_data = train_data_h5[key][:] #(num_trials, len_trial, num_neurons)
valid_data_h5 = h5py.File(f'{rate_dir}/model_runs__valid_{mode}', 'r')
valid_data = valid_data_h5[key][:] #(num_trials, len_trial, num_neurons)
test_data_h5 = h5py.File(f'{rate_dir}/model_runs__test_{mode}', 'r')
test_data = test_data_h5[key][:] #(num_trials, len_trial, num_neurons)
beh=label


# #Jenkins
# batch_size = 100
# optimizer = tf.keras.optimizers.Nadam(0.001)
# rnn_units = 128
# inter_units = 128
# output_shape = 2
# spike_dropout_rate = 0.4
# dropout_rate = 0.7
# L2_kernel = 10
# L2_rec = 10
# L2_inter = 10

# # Chewie
# batch_size = 100
# optimizer = tf.keras.optimizers.Nadam(0.0001)
# rnn_units = 128
# inter_units = 128
# output_shape = 2
# spike_dropout_rate = 0.4
# dropout_rate = 0.7
# L2_kernel = 100
# L2_rec = 100
# L2_inter = 100

# Chewie rate
batch_size = 100
optimizer = tf.keras.optimizers.Nadam(0.0001)
rnn_units = 128
inter_units = 128
output_shape = 2
spike_dropout_rate = 0.4
dropout_rate = 0.7
L2_kernel = 1
L2_rec = 1
L2_inter = 1

params = f"_ds={spike_dropout_rate},do={dropout_rate}"
model = Neural_Model(
    rnn_units,
    inter_units,
    output_shape,
    spike_dropout_rate,
    dropout_rate,
    L2_kernel,
    L2_rec,
    L2_inter,
)

train_losses, valid_losses, train_means, valid_means = [], [], [], []
iterations = 0

if load_model:
    model.load_weights(f"saved_models/{dataset_name[2:]}{beh}_model{params}")
    plot_directions(
        model(train_data), train_behaviours, train_target, dataset_name, "train"
    )
    plot_directions(
        model(test_data), test_behaviours, test_target, dataset_name, "test"
    )
    # r_squared(model(test_data), test_behaviours,test_target,'test', weighted=True)
    r_test_max = rc_squared(test_behaviours, model(test_data), test_target)
    print("Rc: ", r_test_max)
    print("Model loaded")

    with open(f"{dataset_name}{beh}_R2{params}.txt","w") as f:
        f.write(f"{r_test_max*100:.2f}%")

if train:
    max_iter = 3000
    r_test_max = -np.inf
    while iterations <= max_iter:
        iterations += 1
        batch_indices = list(range(train_data.shape[0]))
        batch = np.random.choice(batch_indices, batch_size)
        loss = train_model_behaviour(
            model, optimizer, train_data[batch], train_behaviours[batch]
        )
        train_losses.append(loss.numpy())
        valid_losses.append(
            tf.reduce_mean(
                tf.keras.losses.MSE(valid_behaviours, model(valid_data)).numpy()
            )
        )
        if iterations % 100 == 0:
            train_means.append(np.mean(train_losses))
            valid_means.append(np.mean(valid_losses))
            r_train = rc_squared(train_behaviours, model(train_data), train_target)
            r_test = rc_squared(test_behaviours, model(test_data), test_target)
            print(f"Iteration #{iterations}")
            print(
                "Train/Valid Loss: ",
                np.mean(train_losses),
                " / ",
                np.mean(valid_losses),
            )
            print(
                "Average Test MSE: ",
                tf.reduce_mean(
                    tf.keras.losses.MSE(test_behaviours, model(test_data))
                ).numpy(),
            )
            print("Train/Valid R squared: ", r_train, " / ", r_test)
            r_test_max = max(r_test_max,r_test)

            # plot training curve
            train_losses = []
            valid_losses = []
            if plotting:
                plt.figure()
                plt.plot(train_means, label="Train")
                plt.plot(valid_means, label="Valid")
                plt.legend(loc="upper right")
                plt.yscale("log")
                plt.savefig(f"{dataset_name}{beh}_training.png")
                plt.close()
                plot_directions(
                    model(train_data),
                    train_behaviours,
                    train_target,
                    dataset_name + beh,
                    "train",
                )
                plot_directions(
                    model(test_data),
                    test_behaviours,
                    test_target,
                    dataset_name + beh,
                    "test",
                )
            model.save_weights(f"saved_models/{dataset_name}{beh}_model{params}")
            # print(f'saved_models/{dataset_name}{beh}_model{params}')

            plot_directions(
                model(train_data),
                train_behaviours,
                train_target,
                dataset_name + beh,
                "train",
            )
            plot_directions(
                model(test_data),
                test_behaviours,
                test_target,
                dataset_name + beh,
                "test",
            )

            with open(f"{dataset_name}{beh}_R2{params}.txt","w") as f:
                f.write(str(100*r_test_max))
