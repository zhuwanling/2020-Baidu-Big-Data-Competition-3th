import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
from sklearn.preprocessing import MinMaxScaler
from pandas.plotting import register_matplotlib_converters
from torch import nn, optim
import torch

class MV_LSTM(torch.nn.Module):
    def __init__(self, n_features, opt):
        super(MV_LSTM, self).__init__()
        self.n_features = n_features
        self.seq_len = opt.seq_length
        self.n_hidden = 100  # number of hidden states
        self.n_layers = 6  # number of LSTM layers (stacked)
        self.dropout = nn.Dropout(0.1)

        self.l_lstm = torch.nn.LSTM(input_size=n_features,
                                    hidden_size=self.n_hidden,
                                    num_layers=self.n_layers,
                                    batch_first=True
                                    )
        # according to pytorch docs LSTM output is
        # (batch_size,seq_len, num_directions * hidden_size)
        # when considering batch_first = True
        self.l_linear = torch.nn.Sequential(
            nn.Linear(self.n_hidden* self.seq_len, opt.predict_day*3),
            nn.Linear(opt.predict_day*3, opt.predict_day*2),
            nn.Linear(opt.predict_day*2, opt.predict_day),
        )
        # self.l_linear = torch.nn.Linear(self.n_hidden* self.seq_len, opt.predict_day)
        # self.sigmoid = nn.Sigmoid()

    def init_hidden(self, batch_size):
        # even with batch_first = True this remains same as docs
        hidden_state = torch.zeros(self.n_layers, batch_size, self.n_hidden).cuda()
        cell_state = torch.zeros(self.n_layers, batch_size, self.n_hidden).cuda()

        self.hidden = (hidden_state, cell_state)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()

        lstm_out, self.hidden = self.l_lstm(x, self.hidden)
        # print("lstm_out",lstm_out.shape)
        # lstm_out, self.hidden = self.l_lstm(x)

        # lstm_out(with batch_first = True) is
        # (batch_size,seq_len,num_directions * hidden_size)
        # for following linear layer we want to keep batch_size dimension and merge rest
        # .contiguous() -> solves tensor compatibility error
        x = lstm_out.contiguous().view(batch_size, -1)
        # return self.sigmoid(self.l_linear(x))
        return self.l_linear(x)
