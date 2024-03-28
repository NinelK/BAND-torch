import numpy as np
import h5py
from tqdm import tqdm

import torch
from torch import nn
import pickle

from lfads_torch.benchmark.biRNN_decoder import Decoder, r2_score
from scipy.stats import ttest_ind

BIN_SIZE = 0.01 # seconds, i.e. 10ms

experiments = [
    "Chewie_CO_FF_2016-09-15",
    "Chewie_CO_FF_2016-09-21",
    "Chewie_CO_FF_2016-10-05",
    "Chewie_CO_FF_2016-10-07",
    "Mihili_CO_FF_2014-02-03",  # *
    "Mihili_CO_FF_2014-02-17",  # + (only BL)
    "Mihili_CO_FF_2014-02-18",
    "Mihili_CO_FF_2014-03-07",
]

sel_idxs_all = {
        "Chewie_CO_FF_2016-09-15": [15, 30, 31, 45, 157],
        "Chewie_CO_FF_2016-09-21": [27],
        "Chewie_CO_FF_2016-10-05": [30,44],
        "Chewie_CO_FF_2016-10-07": [1, 27,	28, 35,	144, 157],
        "Mihili_CO_FF_2014-02-03": [13],  
        "Mihili_CO_FF_2014-02-17": [17],
        "Mihili_CO_FF_2014-02-18": [5, 15],
        "Mihili_CO_FF_2014-03-07": [2]
    }

summary_dict = {}   
area = 'all'
n_samples = 10
dropout, spike_dropout_rate = 0.2, 0.3
for short_dataset_name in tqdm(experiments):

    dataset_name = f'{short_dataset_name}_session_vel_{area}_spikes_go'
    loadpath = f'/disk/scratch2/nkudryas/BAND-torch/datasets/{dataset_name}.h5'

    h5file = h5py.File(loadpath, 'r')

    # print(h5file.keys())

    train_data_raw=h5file['train_recon_data'][()].astype(np.float32)
    valid_data_raw=h5file['valid_recon_data'][()].astype(np.float32)
    train_behavior=h5file['train_behavior'][()].astype(np.float32)
    valid_behavior=h5file['valid_behavior'][()].astype(np.float32)
    train_epoch=h5file['train_epoch'][()].astype(np.float32)
    valid_epoch=h5file['valid_epoch'][()].astype(np.float32)
    train_inds=h5file['train_inds'][()].astype(np.float32)
    valid_inds=h5file['valid_inds'][()].astype(np.float32)

    train_target_direction=h5file['train_target_direction'][()].astype(np.float32)
    valid_target_direction=h5file['valid_target_direction'][()].astype(np.float32)
    # print(h5file.keys())
    h5file.close()

    def train_and_get_r2(idxs,
        num_epochs = 1000,
        batch_size = 100):
        '''
        Takes in idxs to ablate
        '''

        train_data = train_data_raw.copy()
        train_data = np.delete(train_data,idxs,axis=-1)
        valid_data = valid_data_raw.copy()
        valid_data = np.delete(valid_data,idxs,axis=-1)

        # train an RNN decoder to predict behavior from neural activity
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        N, M = train_data.shape[-1], train_behavior.shape[-1] # number of neurons, number of behavior dimensions
        T = train_data.shape[1] # number of time bins
        assert M == 2, 'only 2D behavior is expected'
        rnn = Decoder(input_size=N, 
                    rnn_size=128,
                    hidden_size=128, 
                    output_size=M, 
                    seq_len=T, 
                    num_layers=1,
                    spike_dropout_rate = spike_dropout_rate,
                    dropout = dropout).to(device)


        # Loss and optimizer
        loss = nn.MSELoss()
        # higher weight decay for RNN
        optimizer = torch.optim.Adam(rnn.parameters(), lr=0.001, weight_decay=.01)

        #scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.9, verbose=False)

        # Convert numpy arrays to torch tensors
        inputs = torch.from_numpy(train_data).to(device)
        behaviors = torch.from_numpy(train_behavior).to(device)
        test_inputs = torch.from_numpy(valid_data).to(device)
        test_behaviors = torch.from_numpy(valid_behavior).to(device)

        # Train the model
        for epoch in range(num_epochs):

            batch_indices = list(range(inputs.shape[0]))
            batch = torch.from_numpy(np.random.choice(batch_indices, batch_size)).to(device)

            # print(batch)

            # Forward pass
            outputs = rnn(inputs[batch])
            cost = loss(outputs, behaviors[batch]) 

            # Backward and optimize
            optimizer.zero_grad()
            cost.backward()
            optimizer.step()
            scheduler.step(cost)

        # predict on test
        rnn.eval()
        test_outputs = rnn(test_inputs)

        return [r2_score(test_behaviors, test_outputs).item()] +\
            [r2_score(test_behaviors[valid_epoch==e], test_outputs[valid_epoch==e]).item() for e in [0,1,2]]

    sel_idxs =sel_idxs_all[short_dataset_name]
    n_ablate = len(sel_idxs)
    no_ablation, top_score, r_score = [], [], []
    for _ in range(n_samples):

        no_ablation.append(train_and_get_r2([]))

        top_score.append(train_and_get_r2(sel_idxs))

        idxs = np.random.choice(range(train_data_raw.shape[-1]),n_ablate,replace=False)
        r_score.append(train_and_get_r2(idxs))

    print(np.mean(no_ablation,axis=0),np.std(no_ablation,axis=0))
    print(np.mean(top_score,axis=0),np.std(top_score,axis=0))
    print(np.mean(r_score,axis=0),np.std(r_score,axis=0))

    # save results as pkl

    with open(f'./results/abl_r2_{short_dataset_name}.pkl', 'wb') as f:
        pickle.dump([no_ablation, top_score, r_score], f)

    t_top = [ttest_ind(np.array(no_ablation)[:,i], np.array(top_score)[:,i], axis=0, equal_var=False).pvalue for i in range(4)]
    t_r = [ttest_ind(np.array(no_ablation)[:,i], np.array(r_score)[:,i], axis=0, equal_var=False).pvalue for i in range(4)]
    t_between = [ttest_ind(np.array(top_score)[:,i], np.array(r_score)[:,i], axis=0, equal_var=False).pvalue for i in range(4)]

    print(t_top)
    print(t_r)
    print(t_between)

    # only summarize for AD epoch
    summary_dict[short_dataset_name] = {
        'np_ablation_mean': np.mean(no_ablation,axis=0)[2],
        'np_ablation_std': np.std(no_ablation,axis=0)[2],
        'top_score_mean': np.mean(top_score,axis=0)[2],
        'top_score_std': np.std(top_score,axis=0)[2],
        'rand_score_mean': np.mean(r_score,axis=0)[2],
        'rand_score_std': np.std(r_score,axis=0)[2],
        't_top': t_top[2],
        't_rand': t_r[2],
        't_between': t_between[2]
    }            


# save summary
with open(f"./results/abl_R2.csv", "w") as f:
    get_column_names = summary_dict[experiments[0]].keys()
    f.write("Dataset,")
    for key in get_column_names:
        f.write(f"{key},\t")
    f.write('\n')
    for key in summary_dict.keys():
        f.write(f"{key},\t")
        for key2 in get_column_names:
            f.write(f"{summary_dict[key][key2]},\t")
        f.write('\n')