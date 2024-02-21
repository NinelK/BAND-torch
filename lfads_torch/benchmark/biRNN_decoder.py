import torch
import torch.nn as nn

class extract_tensor(nn.Module):
    def forward(self,x):
        tensor, _ = x
        return tensor

class Decoder(nn.Module):
    '''
    1. Dropout some spikes
    2. Run RNN (with batchnorm)
    3. 2 layer MLP
    '''
    def __init__(self, 
                input_size,
                rnn_size, 
                hidden_size, 
                output_size, 
                seq_len, 
                num_layers,
                spike_dropout_rate = 0.4,
                rnn_dropout = 0.,
                dropout = 0.4):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.num_layers = num_layers
        self.input_dropout = nn.Dropout(p=spike_dropout_rate)
        self.rnn = nn.LSTM(input_size,
                          rnn_size, 
                          num_layers=num_layers,
                        #   dropout=rnn_dropout,
                          bidirectional=True,
                          batch_first=True,)
        self.batchnorm = nn.BatchNorm1d(seq_len, track_running_stats=False)
        self.fc = nn.Linear(2*rnn_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):

        # Set initial hidden and cell states
        # h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).cuda()
        # c0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).cuda()
        
        input = self.input_dropout(x)
        # Forward propagate
        out, _ = self.rnn(input) #, (h0,c0))
        out = nn.ReLU()(out)
        out = self.batchnorm(out)
  
        out = self.fc(out)
        out = nn.ReLU()(out)
        out = self.batchnorm(out)
        out = self.dropout(out)
        out = self.fc2(out)


        return out
    
def r2_score(y_true, y_pred):
    SS_res =  torch.sum((y_true - y_pred)**2, axis=[0,1])
    SS_tot = torch.sum((y_true - torch.mean(y_true, axis=[0,1]))**2, axis=[0,1])
    return torch.mean(1 - SS_res/SS_tot)