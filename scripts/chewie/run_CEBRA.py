import numpy as np
import h5py
from tqdm import tqdm
from cebra import CEBRA, KNNDecoder
import sklearn.metrics

max_iterations = 10000
batch_size = 512
output_dimension = 100
lr = 0.0001
T = 1

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

def r2_score(y_true, y_pred):
    y_true = y_true.reshape(-1, y_true.shape[-1])
    y_pred = y_pred.reshape(-1, y_pred.shape[-1])
    SS_res =  np.sum((y_true - y_pred)**2, axis=0)
    SS_tot = np.sum((y_true - np.mean(y_true, axis=0))**2, axis=0)
    return np.mean(1 - SS_res/SS_tot)

def decoding_pos(embedding_train, embedding_test, label_train, label_test, n_neighbors=400):
   pos_decoder = KNNDecoder(n_neighbors=n_neighbors, metric="cosine")
   
   pos_decoder.fit(embedding_train, label_train)
#    print('Trained')
   
   pos_pred_test = pos_decoder.predict(embedding_test)
#    print('Predicted test')
   pos_pred_train = pos_decoder.predict(embedding_train)
#    print('Predicted train')
   
   train_score = sklearn.metrics.r2_score(label_train, pos_pred_train)
   test_score = sklearn.metrics.r2_score(label_test, pos_pred_test)
   pos_train_err = np.median(abs(pos_pred_train - label_train))
   pos_test_err = np.median(abs(pos_pred_test - label_test))
   
   return {'train_score': train_score, 
           'train_err': pos_train_err, 
           'test_score': test_score, 
           'test_err': pos_test_err,
           'train_pred': pos_pred_train,
           'test_pred': pos_pred_test}

def fine_tune_n_neighbors(cebra_pos_train, label_train, cebra_pos_test, label_test):
    '''Fine-tune the KNN decoder, according to Cebra paper
    'For CEBRA we used kNN regression, and the number of neighbours k was again searched over [1, 2500].'
    '''
    nn = [1,100,200,300,400,500,1000,1500,2000,2500]

    scores = []
    for n_neighbors in nn:
        cebra_pos_decode = decoding_pos(cebra_pos_train, cebra_pos_test, label_train, label_test, n_neighbors=n_neighbors)
        scores.append(cebra_pos_decode['test_score'])
    # plt.plot(nn,scores)
    # plt.scatter(nn,scores)

    n_neighbors = nn[np.argmax(scores)]
    return n_neighbors

def save_results(f,area,train_outputs, test_outputs):
    if f'train_{area}_cebra_pred' in f:
        del f[f'train_{area}_cebra_pred']
    if f'test_{area}_cebra_pred' in f:
        del f[f'test_{area}_cebra_pred']
    f.create_dataset(f'train_{area}_cebra_pred', data=train_outputs)
    f.create_dataset(f'test_{area}_cebra_pred', data=test_outputs)

summary_dict = {}   
for short_dataset_name in tqdm(experiments):

    summary_dict[short_dataset_name] = {}

    for area in ['all','PMd','M1']:

        dataset_name = f'{short_dataset_name}_session_vel_{area}_spikes_go'
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
            
        neural_pos_train, neural_pos_test = np.concatenate(train_data), np.concatenate(valid_data)
        label_train, label_test = np.concatenate(train_behavior), np.concatenate(valid_behavior)

        cebra_pos_model = CEBRA(model_architecture='offset10-model',
                        batch_size=batch_size,
                        learning_rate=lr,
                        temperature=T,
                        output_dimension=output_dimension,
                        max_iterations=max_iterations,
                        distance='cosine',
                        conditional='time_delta',
                        device='cuda_if_available',
                        verbose=True,
                        time_offsets=10)
        
        # train on train, predict for both
        cebra_pos_model.fit(neural_pos_train,label_train)
        cebra_pos_train = cebra_pos_model.transform(neural_pos_train)
        cebra_pos_test = cebra_pos_model.transform(neural_pos_test)

        # # fine-tune n_neighbors (use validation set, then fix)
        # n_neighbors = fine_tune_n_neighbors(cebra_pos_train, label_train, cebra_pos_test, label_test)
        # print(f"Fine-tuned n_neighbors: {n_neighbors} for {short_dataset_name} {area}")

        n_neighbors = 400

        cebra_pos_decode = decoding_pos(cebra_pos_train, cebra_pos_test, label_train, label_test, n_neighbors=n_neighbors)
        train_outputs = cebra_pos_decode['train_pred'].reshape(train_behavior.shape)
        test_outputs = cebra_pos_decode['test_pred'].reshape(valid_behavior.shape)

        print(cebra_pos_decode['train_score'], cebra_pos_decode['test_score'])

        final_r2 = r2_score(valid_behavior, test_outputs).item()
        summary_dict[short_dataset_name][f"{area}_R2_all"] = np.round(100*final_r2,1) 

        for e, epoch_name in enumerate(['BL','AD','WO']):
            final_r2 = r2_score(valid_behavior[valid_epoch==e], test_outputs[valid_epoch==e]).item()
            summary_dict[short_dataset_name][f"{area}_R2_{epoch_name}"] = np.round(100*final_r2,1)       

        # save predictions
        results_path = f'./results/{short_dataset_name}.h5'
        with h5py.File(results_path, 'a') as f:
            save_results(f,area,train_outputs, test_outputs)

# save summary
with open("./results/cebra_decoder_R2.csv", "w") as f:
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