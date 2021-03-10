import pandas as pd
from numpy import array
import numpy as np
from sklearn.preprocessing import MinMaxScaler
def gettest():
    #read testing data##############################################################################################
    df2 = pd.read_csv('./lstm/data/lstm.csv',
                      header=None,
                      names=['cityname', 'region_ID', 'data','infectnum', 'infectnum_mean', 'infectnum_max', 'infectnum_sum',
                        'infectnum_min', 'infectnum_median', 'infectnum_std', 'infectnum_skew', 'infectnum_kurt',
                        'infectnum_quantile_25', 'infectnum_quantile_75', 'migration', 'density', 'transfer'])
    # df2.head()
    # df2.info()
    # is_indonesia =  (df2['cityname']=='E')
    # testing data filtering#########################################################################################
    test_x = df2[['cityname', 'region_ID', 'data', 'infectnum', 'infectnum_mean', 'infectnum_max', 'infectnum_sum',
                  'infectnum_min', 'infectnum_median', 'infectnum_std', 'infectnum_skew', 'infectnum_kurt',
                  'infectnum_quantile_25', 'infectnum_quantile_75', 'migration', 'density', 'transfer',
                  ]]
    # test_x = df2[(is_indonesia)][['region_ID','confirmed']]
    # test_y = df2[(is_indonesia)][['region_ID','day', 'confirmed']]
    # test_y.day = pd.to_datetime(test_y.day, format='%Y%m%d', errors='ignore')
    # test_y.set_index('day', inplace=True)
    return test_x

def getdate(opt):
    #read training data#############################################################################################
    df = pd.read_csv('./lstm/data/lstm.csv',header=None,
                      names=['cityname', 'region_ID', 'data', 'infectnum', 'infectnum_mean', 'infectnum_max', 'infectnum_sum',
                  'infectnum_min', 'infectnum_median', 'infectnum_std', 'infectnum_skew', 'infectnum_kurt',
                  'infectnum_quantile_25', 'infectnum_quantile_75', 'migration', 'density', 'transfer'])
    # df.head()
    # df.info()
    #training data filtering#########################################################################################
    data=df[df.cityname.isin(['B','C','D','E','F','G','H','I','J','K'])][[ 'infectnum', 'infectnum_mean', 'infectnum_max', 'infectnum_sum',
                  'infectnum_min', 'infectnum_median', 'infectnum_std', 'infectnum_skew', 'infectnum_kurt',
                  'infectnum_quantile_25', 'infectnum_quantile_75', 'migration', 'density', 'transfer',
                  ]]
    # input splitting################################################################################################
    X, Y = split_sequences(data.values, opt)
    # normalization##################################################################################################
    alld = np.concatenate((X, Y), 1)
    alld = alld.reshape(alld.shape[0] * alld.shape[1], alld.shape[2])
    scaler = MinMaxScaler()
    scaler.fit(alld)
    X = [scaler.transform(x) for x in X]
    y = [scaler.transform(y) for y in Y]
    X = np.array(X)
    y = np.array(y)[:, :, 0]
    return X,y,scaler

# split a multivariate sequence into samples
def split_sequences(sequences, opt):
    X, y = list(), list()
    for i in range(0, len(sequences), opt.time_length):
        # find the end of this pattern
        end_ix = i + opt.time_length
        # check if we are beyond the dataset
        if i != 0 and end_ix > len(sequences):
            break
        sequences[i:end_ix, 0] = np.insert(np.diff(sequences[i:end_ix, 0]), 0, 0)
        # print(sequences[i:end_ix, 0])
        # 滑动窗口
        for j in range(i,i+(opt.time_length - (opt.predict_length+opt.seq_length)+1)):
            # print(j)
            seq_x, seq_y = sequences[j: j+opt.seq_length], sequences[j+opt.seq_length:j+opt.seq_length+opt.predict_length]
            # print(seq_y[:,0])
            X.append(seq_x)
            y.append(seq_y)
            # print('x：',seq_x[:,0])
            # print('y:',seq_y[:,0])
    return array(X), array(y)