import torch

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .model import MV_LSTM
from .data import getdate,gettest
import pickle
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.preprocessing import MinMaxScaler
import argparse
from numpy import array

parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument('--n_features', type=int, default=14, help="this is number of parallel inputs")
parser.add_argument('--train_episodes', type=int, default=10000, help='train epoch')
parser.add_argument('--batch_size', type=int, default=100, help='batch_size')
parser.add_argument('--learn_rate', type=float, default=1e-4, help='Learning Rate. Default=0.01')
parser.add_argument('--time_length', type=int, default=60, help='all time length')
parser.add_argument('--predict_length', type=int, default=5, help='predict time length')
parser.add_argument('--seq_length', type=int, default=20, help='train time length')
parser.add_argument('--save_file', type=str, default='./model/', help='save model file')
parser.add_argument('--prefix', type=str, default='testname', help='prefix')
parser.add_argument('--pretrained', type=bool, default=True, help='is load weight')
parser.add_argument('--pretrained_sr', default='testname_epoch_10000.pth', help='sr pretrained base model')
parser.add_argument('--predict_day', type=int, default=5, help='predict day')
opt = parser.parse_args()

#get data
X, y, scaler = getdate(opt) #训练数据
test_x= gettest() #测试数据test_y

#build model
mv_net = MV_LSTM(opt.n_features,opt).cuda()
criterion = torch.nn.MSELoss() # reduction='sum' created huge loss value
optimizer = torch.optim.Adam(mv_net.parameters(), lr=opt.learn_rate)

def checkpoint():
    model_out_path = opt.save_file+opt.prefix+"_epoch_{}.pth".format(opt.train_episodes)
    torch.save(mv_net.state_dict(), model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))


def train():
    mv_net.train()
    for t in range(opt.train_episodes):

        for b in range(0,len(X),opt.batch_size):

            p = np.random.permutation(len(X))
            inpt = X[p][b:b+opt.batch_size,:,:]
            target = y[p][b:b+opt.batch_size,:]


            x_batch = torch.tensor(inpt,dtype=torch.float32).cuda()
            y_batch = torch.tensor(target,dtype=torch.float32).cuda()

            mv_net.init_hidden(x_batch.size(0))

            output = mv_net(x_batch)

            # all_batch=torch.cat((x_batch[:,:,0], y_batch), 1)
            # print("inout:",scaler.inverse_transform(x_batch[0].cpu().numpy())[:,0])
            # print("output",output)
            # print("stander",y_batch)
            loss = 10000*criterion(output.view(-1), y_batch.view(-1))  #all_batch

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        print('step : ' , t , 'loss : ' , loss.item())

def run():
    if opt.pretrained:
        model_name = os.path.join(opt.save_file + opt.pretrained_sr)
        if os.path.exists(model_name):
            #model= torch.load(model_name, map_location=lambda storage, loc: storage)
            mv_net.load_state_dict(torch.load(model_name, map_location=lambda storage, loc: storage))
            print('Pre-trained SR model is loaded.')
    else:
        train()
        checkpoint()
    #evaluation#########################################################################################################
    groupdate = test_x.groupby(['cityname','region_ID'])
    resolut = []
    location_infect=3
    for name ,data in groupdate:
        print(name)
        data = data.copy().reset_index(drop = True)
        data['confirmed'] = np.insert(np.diff(data.values[0:len(data),location_infect]),0,0)
        data2x = data[['infectnum', 'infectnum_mean', 'infectnum_max', 'infectnum_sum',
                      'infectnum_min', 'infectnum_median', 'infectnum_std', 'infectnum_skew', 'infectnum_kurt',
                      'infectnum_quantile_25', 'infectnum_quantile_75', 'migration', 'density', 'transfer',]]
        data2x=scaler.transform(data2x)
        # data2x = data2x[:,0]
        X_test = np.expand_dims(data2x, axis=0)
        # print(scaler.inverse_transform(X_test[:,-opt.seq_length:,:][0])[:,0])
        # print(X_test.shape)
        # X_test = np.expand_dims(X_test, axis=2)
        mv_net.init_hidden(1)
        lstm_out = mv_net(torch.tensor(X_test[:,-opt.seq_length:,:],dtype=torch.float32).cuda())
        lstm_out=lstm_out.reshape(1,opt.predict_day,1).cpu().data.numpy()
        #将标准化化的数据还原，为什么要复制为n_features？
        actual_predictions = scaler.inverse_transform(np.tile(lstm_out, (1, 1,opt.n_features))[0])[:,0]
        # print(actual_predictions)
        predict_diff = pd.DataFrame(actual_predictions)
        predict_diff.rename(columns={ 0:'diff'},inplace=True)
        predict_diff['cityname'] =data['cityname']
        predict_diff['region_ID'] = data['region_ID']
        # predict_diff['date']  = pd.to_datetime(data['date']) + pd.DateOffset(days=opt.predict_length)
        resolut.append(predict_diff)
    # save prediction
    resolut = pd.concat(resolut)
    resolut.to_csv('./submission/resolute.csv',index=False,header=None)

    #visualization####################################################################################################
    # fig, ax = plt.subplots()
    # plt.title('Days vs Confirmed Cases Accumulation')
    # plt.ylabel('Confirmed')
    #
    # left, width = .25, .5
    # bottom, height = .25, .5
    # right = left + width
    # top = bottom + height
    #
    # # print (test_y.index)
    # date_list=pd.date_range(start=test_y.index[0],end=test_y.index[-1])
    #
    # plt.axvline(x=np.array(date_list)[opt.seq_length], color='r', linestyle='--')
    #
    # ax.text(0.2*(left+right), 0.8*(bottom+top), 'input sequence',
    #         horizontalalignment='left',
    #         verticalalignment='center',
    #         fontsize=10, color='red',
    #         transform=ax.transAxes)
    # ax.text(0.0125*(left+right), 0.77*(bottom+top), '___________________________',
    #         horizontalalignment='left',
    #         verticalalignment='center',
    #         fontsize=20, color='red',
    #         transform=ax.transAxes)
    #
    #
    # # print(actual_predictions)
    # sumpred=np.cumsum(actual_predictions)
    # # print(sumpred)
    # # print (test_y.values.shape)
    # print (sqrt(mean_squared_error(test_y.confirmed,sumpred)))
    # # plt.plot(test_y.values[-opt.seq_length:],np.cumsum(test_x.confirmed.values[-opt.seq_length:]))
    # plt.plot(np.array(date_list),sumpred,label='Prediction')
    # plt.plot(np.array(date_list),test_y.confirmed,label='Actual')
    # plt.xticks(rotation=90)
    # fig.autofmt_xdate()
    # plt.legend(loc=2)
    # plt.show()

