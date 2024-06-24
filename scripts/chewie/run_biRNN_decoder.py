import numpy as np
import h5py
import torch
from torch import nn
from tqdm import tqdm

from lfads_torch.benchmark.biRNN_decoder import Decoder, r2_score

num_epochs = 1000 # 1000 is enough
batch_size = 100

data_type = 'spikes' # 'spikes' or 'MUA'

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

def save_results(f,area,train_outputs, test_outputs):
    
    if data_type == 'spikes':
        p = ''
    elif data_type == 'MUA':
        p = '_MUA'
    else:
        raise ValueError('data_type should be spikes or MUA')
    
    if f'train_{area}{p}_birnn_pred' in f:
        del f[f'train_{area}{p}_birnn_pred']
    if f'test_{area}{p}_birnn_pred' in f:
        del f[f'test_{area}{p}_birnn_pred']
    f.create_dataset(f'train_{area}{p}_birnn_pred', data=train_outputs)
    f.create_dataset(f'test_{area}{p}_birnn_pred', data=test_outputs)

summary_dict = {}   
for short_dataset_name in tqdm(experiments):

    summary_dict[short_dataset_name] = {}

    for area in ['all','PMd','M1']:

        dataset_name = f'{short_dataset_name}_session_vel_{area}_{data_type}_go'
        loadpath = f'/disk/scratch2/nkudryas/BAND-torch/datasets/{dataset_name}.h5'

        with h5py.File(loadpath, 'r') as h5file:

            train_data=h5file['train_recon_data'][()].astype(np.float32)
            valid_data=h5file['valid_recon_data'][()].astype(np.float32)
            train_behavior=h5file['train_behavior'][()].astype(np.float32)
            valid_behavior=h5file['valid_behavior'][()].astype(np.float32)
            train_epoch=h5file['train_epoch'][()].astype(np.float32)
            valid_epoch=h5file['valid_epoch'][()].astype(np.float32)
            train_inds=h5file['train_inds'][()].astype(np.float32)
            valid_inds=h5file['valid_inds'][()].astype(np.float32)

            train_target_direction=h5file['train_target_direction'][()].astype(np.float32)
            valid_target_direction=h5file['valid_target_direction'][()].astype(np.float32)
            
        # train an RNN decoder to predict behavior from neural activity
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        N, M = train_data.shape[-1], train_behavior.shape[-1] # number of neurons, number of behavior dimensions
        T = train_data.shape[1] # number of time bins
        assert M == 2, 'only 2D behavior is expected'
        
        dropout, spike_dropout_rate = 0.2, 0.3
        rnn = Decoder(input_size=N, 
                    rnn_size=128,
                    hidden_size=128, 
                    output_size=M, 
                    seq_len=T, 
                    num_layers=1,
                    dropout=dropout,
                    spike_dropout_rate=spike_dropout_rate).to(device)

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

            # print (f'Epoch [{epoch+1}/{num_epochs}], Loss: {cost.item():.4f}, Test Loss: {test_cost.item():.4f}')
            # print epoch and R2
            if (epoch+1) % 200 == 0:
                # predict on test
                rnn.eval()
                train_outputs = rnn(inputs)
                train_cost = loss(train_outputs, behaviors) 
                test_outputs = rnn(test_inputs)
                test_cost = loss(test_outputs, test_behaviors)
                rnn.train()
                
                print (f'Epoch [{epoch+1}/{num_epochs}], Loss: {train_cost.item():.4f}, Test Loss: {test_cost.item():.4f}, R2: {r2_score(test_behaviors, test_outputs).item():.4f}')
                       
        final_r2 = r2_score(test_behaviors, test_outputs).item()
        summary_dict[short_dataset_name][f"{area}_R2_all"] = np.round(100*final_r2,1) 

        for e, epoch_name in enumerate(['BL','AD','WO']):
            final_r2 = r2_score(test_behaviors[valid_epoch==e], test_outputs[valid_epoch==e]).item()
            summary_dict[short_dataset_name][f"{area}_R2_{epoch_name}"] = np.round(100*final_r2,1)       

        # save predictions
        results_path = f'./results/{short_dataset_name}.h5'
        with h5py.File(results_path, 'a') as f:
            save_results(f,area,train_outputs.cpu().detach().numpy(), test_outputs.cpu().detach().numpy())

# save summary
with open("./results/biRNN_decoder_R2.csv", "w") as f:
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