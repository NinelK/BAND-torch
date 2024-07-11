import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

class Neural_Model(tf.keras.Model):
    def __init__(
        self,
        gru_units,
        inter_units,
        output_size,
        spike_dropout_rate,
        dropout_rate,
        L2_kernel,
        L2_rec,
        L2_inter,
    ):
        super(Neural_Model, self).__init__()
        self.drop_spikes = tf.keras.layers.Dropout(spike_dropout_rate)
        self.gru = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(
                gru_units,
                kernel_regularizer=tf.keras.regularizers.L2(L2_kernel),
                recurrent_regularizer=tf.keras.regularizers.L2(L2_rec),
                return_sequences=True,
            )
        )
        self.batch_norm = tf.keras.layers.BatchNormalization()
        self.inter = tf.keras.layers.Dense(
            inter_units,
            activation="relu",
            kernel_regularizer=tf.keras.regularizers.L2(L2_inter),
        )
        self.batch_norm_inter = tf.keras.layers.BatchNormalization()
        self.drop_layer = tf.keras.layers.Dropout(dropout_rate)
        self.readout = tf.keras.layers.Dense(output_size, use_bias=False)

    def call(self, neural_batch, get_cell_activations=False):
        neural_batch = self.drop_spikes(tf.cast(neural_batch, tf.float32))
        neural_act = self.batch_norm(self.gru(tf.cast(neural_batch, tf.float32)))
        if get_cell_activations:
            return neural_act
        self.out = self.drop_layer(self.batch_norm_inter(self.inter(neural_act)))
        self.behaviour_pred = self.readout(self.out)

        return self.behaviour_pred


def train_model_behaviour(model, optimizer, neural_batch, gt_behaviour):
    with tf.GradientTape() as tape:
        predictions = model(neural_batch, training=True)
        pred_loss = tf.reduce_mean(tf.keras.losses.MSE(gt_behaviour, predictions))
    gradients = tape.gradient(pred_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return pred_loss


def rc_squared(
    behaviour: tf.Tensor, b: tf.Tensor, condition: np.ndarray, return_all=False
):
    """
    Calculates R2_c [directions]
    Inputs
    ------
    behaviour: True behavior [trials x time x behaviours]
    b        : Predicted behavior [trials x time x behaviours]
    condition: Some labels for experimental conditions (e.g. reach directions)
    """
    unique_dirs = np.unique(condition)
    R2_t = np.zeros((len(unique_dirs)))  # directions
    for i, d in enumerate(unique_dirs):
        unexplained_error = tf.reduce_sum(
            tf.square(behaviour[condition == d] - b[condition == d])
        ).numpy()
        total_error = (
            tf.reduce_sum(
                tf.square(
                    behaviour[condition == d]
                    - tf.reduce_mean(behaviour[condition == d], axis=[0])
                )
            ).numpy()
            + 1e-10
        )
        if len(b[condition == d]) > 1:
            R2_t[i] = 1 - unexplained_error / total_error

    if return_all is True:
        return R2_t
    else:
        return R2_t.mean()

def np_rc_squared(
    behaviour: np.array, b: np.array, condition: np.ndarray, return_all=False
):
    """
    Calculates R2_c [directions]
    Inputs
    ------
    behaviour: True behavior [trials x time x behaviours]
    b        : Predicted behavior [trials x time x behaviours]
    condition: Some labels for experimental conditions (e.g. reach directions)
    """
    unique_dirs = np.unique(condition)
    R2_t = np.zeros((len(unique_dirs)))  # directions
    for i, d in enumerate(unique_dirs):
        unexplained_error = np.sum((behaviour[condition == d] - b[condition == d]) ** 2)
        total_error = (
            np.sum(
                (behaviour[condition == d] - np.mean(behaviour[condition == d], axis=0))
                ** 2
            )
            + 1e-10
        )
        if len(b[condition == d]) > 1:
            R2_t[i] = 1 - unexplained_error / total_error

    if return_all is True:
        return R2_t
    else:
        return R2_t.mean()


def plot_directions(
    predicted_behaviours,
    ground_truth_behaviours,
    ground_truth_directions,
    dataset_name,
    label,
):
    dir_index = [
        sorted(set(ground_truth_directions)).index(i) for i in ground_truth_directions
    ]
    colors=plt.cm.hsv(ground_truth_directions / (2*np.pi) + 0.5)
    # colors = plt.cm.nipy_spectral(np.arange(8) / 8)
    plt.figure(figsize=(5, 5))
    for t in range(0, predicted_behaviours.shape[0], 3):
        plt.plot(
            ground_truth_behaviours[t, :, 0],
            ground_truth_behaviours[t, :, 1],
            color=colors[dir_index[t]],
            alpha=0.5,
            ls="--",
        )
        plt.plot(
            predicted_behaviours[t, :, 0],
            predicted_behaviours[t, :, 1],
            color=colors[dir_index[t]],
            alpha=0.5,
            lw=2,
        )
    plt.title("Behaviour (whole time interval)")
    plt.legend(("true", "model"))
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.xticks(())
    plt.yticks(())
    plt.savefig(f"{dataset_name}_rec_{label}.png")
    plt.close()


def calc_rate(spike_data, w=2):
    # kernel
    f = np.linspace(-4 * w, 4 * w, 8 * w - 1)
    K = 1 / (np.sqrt(2 * np.pi) * w) * np.exp(-(f**2) / (2 * w**2))

    rate_data = np.zeros_like(spike_data).astype("float")
    for i in range(spike_data.shape[0]):
        for j in range(spike_data.shape[-1]):
            rate_data[i, :, j] = np.convolve(
                spike_data[i, :, j], K, mode="same"
            )  # (num_trials, len_trial, num_neurons)
    return rate_data


def direction_average_rate(rate, targets):
    assert (
        targets.shape[0] == rate.shape[0]
    ), "Number of trials should be the same in rates and targets"
    unique_targets = np.unique(targets)
    dir_avg_rates = {}
    for target in unique_targets:
        mask = targets == target
        dir_avg_rates[target] = rate[mask].mean(0)
    return dir_avg_rates


def fill_in_average_rate(dir_avg_rates: dict, targets, data_shape):
    filled_rate = np.zeros(data_shape, dtype="float")
    unique_targets = np.unique(targets)
    for target in unique_targets:
        mask = targets == target
        filled_rate[mask] = dir_avg_rates[target]
    return filled_rate
