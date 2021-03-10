import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
from sklearn.metrics import  mean_squared_log_error
import datetime
from tqdm import tqdm

avgmigration = {'BA': [0.268893, 0.278251, 0.279033, 0.312693, 0.289101, 0.317763, 0.282647],
                'CA': [0.152966, 0.141885, 0.141787, 0.145828, 0.146267, 0.138926, 0.095909],
                'DA': [0.281869, 0.269060, 0.287152, 0.299917, 0.286138, 0.287680, 0.226503],
                'EA': [0.069098, 0.061522, 0.064633, 0.064902, 0.066277, 0.061781, 0.048465],
                'AB': [0.256419, 0.265502, 0.300005, 0.315109, 0.297525, 0.331744, 0.307768],
                'CB': [0.011443, 0.009877, 0.011687, 0.010965, 0.010562, 0.010843, 0.010352],
                'DB': [0.019678, 0.021584, 0.021134, 0.022370, 0.022476, 0.020914, 0.020650],
                'EB': [0.004385, 0.003974, 0.004351, 0.004110, 0.004425, 0.004325, 0.004169],
                'AC': [0.101531, 0.082550, 0.089771, 0.088970, 0.085610, 0.100532, 0.089672],
                'BC': [0.011308, 0.010589, 0.009993, 0.009035, 0.009776, 0.009995, 0.008743],
                'DC': [0.064417, 0.064946, 0.062949, 0.061194, 0.060398, 0.062246, 0.055507],
                'EC': [0.074979, 0.078030, 0.081791, 0.086364, 0.083897, 0.079726, 0.072333],
                'AD': [0.204827, 0.205227, 0.221963, 0.217700, 0.215062, 0.229749, 0.205794],
                'BD': [0.020293, 0.019753, 0.020130, 0.019898, 0.019329, 0.019510, 0.020169],
                'CD': [0.071377, 0.064978, 0.070155, 0.072553, 0.070549, 0.068240, 0.064930],
                'ED': [0.080028, 0.085126, 0.088498, 0.094469, 0.090239, 0.086924, 0.082820],
                'AE': [0.049674, 0.043264, 0.045341, 0.044346, 0.045091, 0.050614, 0.044371],
                'BE': [0.004433, 0.004190, 0.003901, 0.003999, 0.004091, 0.003812, 0.003715],
                'CE': [0.085984, 0.083894, 0.090516, 0.093224, 0.090683, 0.090045, 0.082571],
                'DE': [0.084931, 0.086124, 0.091071, 0.094052, 0.091289, 0.088074, 0.081129], }
#返回{minute_series: [basefature+travvel_time]}
def bucket_data(lines):
    bucket = {}
    for line in lines:
        time_series = line[-2]
        bucket[time_series] = []
    for line in lines:
        time_series, y1 = line[-2:]
        line = np.delete(line, -2, axis=0)
        bucket[time_series].append(line)
    return bucket

def feature_vis(regressor, train_feature):
    importances = regressor.feature_importances_
    indices = np.argsort(importances)[::-1]
    selected_features = [train_feature[e] for e in indices]
    plt.figure(figsize=(20, 10))
    plt.title("train_feature importances")
    plt.bar(range(len(train_feature)), importances[indices],
            color="r", align="center")
    plt.xticks(range(len(selected_features)), selected_features, rotation=70)
    plt.show()

def cross_valid(regressor, bucket, lagging):
    valid_loss = []
    last = [[] for i in range(len(bucket[list(bucket.keys())[0]]))]
    sort_key = sorted(bucket.keys(), key=float)
    for day_series in sort_key:
        if int(day_series) in range(int(sort_key[0]), int(sort_key[0]) + lagging ):
            last = np.concatenate((last, np.array(bucket[day_series], dtype=float)[:, -1].reshape(-1, 1)), axis=1)
        else:
            batch = np.array(bucket[day_series], dtype=float)
            y = batch[:, -1]
            batch = np.delete(batch, -1, axis=1)
            batch = np.concatenate((batch, last), axis=1)
            y_pre = regressor.predict(batch)
            last = np.delete(last, 0, axis=1)
            last = np.concatenate((last, y_pre.reshape(-1, 1)), axis=1)
            loss = math.sqrt(mean_squared_log_error(y_pre,y))
            valid_loss.append(loss)
    # print 'day: %d loss: %f' % (int(day), day_loss)
    return np.mean(valid_loss)


def Wow_trend(now,last,day,log):
    Wow_ = now.copy(deep=True)
    last_list = last['infectnum'].tolist()
    now_list = now['infectnum'].tolist()
    diff_list=[]
    # print(len(last_list))
    # print(len(now_list))
    for i in range(last.shape[0]):
        if (log == True):
            diff = math.expm1(now_list[i]) / (math.expm1(last_list[i])+1)
            # diff = (math.expm1(now_list[i]) - math.expm1(last_list[i]))
        else:
            diff = (now_list[i] - last_list[i])
        diff_list.append(diff)

    Wow_['infectnum'] = diff_list
    return Wow_
def Sum_trend(now_date,test_df,df_history_read,day,log):
    sum = test_df.copy()
    sum_now_tolist = test_df['infectnum'].values.tolist()
    for cc in range(len(sum_now_tolist)):
        if log == True:
            sum_now_tolist[cc] = math.exp(sum_now_tolist[cc]) - 1
        else:
            sum_now_tolist[cc] = sum_now_tolist[cc]
    for cnt in range(1, day + 1):
        last_day_ = now_date - datetime.timedelta(cnt)
        df_history_read_list = df_history_read.loc[df_history_read['data'] == last_day_, 'infectnum'].values.tolist()
        for cc in range(len(df_history_read_list)):
            if log == True:
                df_history_read_list[cc] = math.exp(df_history_read_list[cc]) - 1
            else:
                df_history_read_list[cc] = df_history_read_list[cc]
        for ccc in range(len(sum_now_tolist)):
            sum_now_tolist[ccc] = sum_now_tolist[ccc] + df_history_read_list[ccc]
    for ccc in range(len(sum_now_tolist)):
        if sum_now_tolist[ccc] <= 0:
            sum_now_tolist[ccc] = 0
            sum_now_tolist[ccc] = math.log1p(sum_now_tolist[ccc])
        if sum_now_tolist[ccc] > 0:
            sum_now_tolist[ccc] = math.log1p(sum_now_tolist[ccc])
    sum['infectnum'] = sum_now_tolist
    return sum

def Migration_trend(test_df, lasttime, day,nowday):
    # 做今日城市日增长汇总infectsum
    tody = test_df.copy()
    infectsum = tody.groupby(['cityname', 'data']).agg({'predicted': 'sum'}).reset_index()

    infectsum.rename(columns={'predicted': 'infectdaysum'}, inplace=True)

    df_m = tody[['cityname', 'region_ID', 'data']]
    df1 = df_m.values
    df2 = []
    for date in df1:
        date = list(date)
        # 找到改天推移day日该城市的所有入口城市,5代表城市数量
        suminfect = 0
        for i in range(5):
            startcity = chr(ord('A') + i)
            # 自己到自己跳过
            if (chr(ord('A') + i) == date[0]): continue
            keyworld = startcity + date[0]
            rate = avgmigration[keyworld][lasttime.weekday()]
            num = list(infectsum.loc[infectsum['cityname'] == startcity].values[0])[2]
            suminfect = suminfect + rate * num
        date.append(math.log1p(suminfect))
        df2.append(date)
    df2 = pd.DataFrame(df2)
    df2.rename(columns={0: 'cityname', 1: 'region_ID', 2: 'data', 3: 'migration'}, inplace=True)
    return df2['migration']

def kurt(df):
    return df.kurt()
#分位数
def quantile_25(df):
    return df.quantile(q=0.25)
def quantile_75(df):
    return df.quantile(q=0.75)


# ------------------------------------------------Submission ---------------------------------------------
def submission(train_feature, regressor, df, file,lag_num,log):
    #存历史数据
    df['int_data'] =df['data']
    df['predicted'] = df['infectnum']
    files = './code78_128/dataset/data_processed/history_data.csv'
    df.to_csv(files,index=False)#存入60天数据 -0629号
    #构建6.15日lagging特征
    test_df = df.loc[((df['data'].dt.year == 2120) & (df['data'].dt.month == 6)
                      & (df['data'].dt.day== 29))].copy()
    test_df.reset_index(inplace=True,drop=True)

    col_name = ['infectnum'] #,'infectnum_WoW','infectnum_Sum','migration'
    #lag循环直至lag2
    if lag_num!=1:
        for num in range(lag_num,1,-1):
            for col_ in col_name:
                former = col_ + '_lagging{}'.format(num)
                latter = col_ + '_lagging{}'.format(num-1)
                test_df[[former]] = test_df[[latter]]

    #lag为1的时候与特征变量进行交换
    num = 1
    for col_ in col_name:
        former = col_ + '_lagging{}'.format(num)
        latter = col_
        test_df[[former]] = test_df[[latter]]

    with open(file, 'w'):
        pass

    column_sort = test_df.columns.values
    for i in tqdm(range(30)):
        #预测构建好的特征对应感染数
        test_X = test_df[train_feature]
        y_prediction = regressor.predict(test_X.values)
        test_df['infectnum'] = y_prediction
        test_df['predicted'] = y_prediction
        test_df['data'] = test_df['data'] + pd.DateOffset(days=1)
        #更新今日非lagging特征
        df_history_read = pd.read_csv(files, delimiter=',', parse_dates=['data'])
        day = 4
        #WoW
        # for k in [4]:
        #     now = test_df.copy(deep=True)
        #     last = df_history_read.loc[
        #         df_history_read['data'] == (datetime.datetime(2120, 6, 29) + datetime.timedelta(i - k))].copy(deep=True)
        #     Wow = Wow_trend(now, last, k,log)
        #     test_df['infectnum_WoW%d'%k] = Wow['infectnum'].copy(deep=True)
        # #Sum
        # for k in [4]:
        #     now_date = datetime.datetime(2120, 6, 29) + datetime.timedelta(i + 1)
        #     Sum = Sum_trend(now_date, test_df.copy(), df_history_read.copy(), k,log)
        #     test_df['infectnum_Sum%d'%k] = Sum['infectnum'].copy(deep=True)
        #transfer
        # tranfer = tranfer_trend(test_df.copy())
        #migration
        # day = 10
        # lasttime =  datetime.datetime(2120, 6, 29) + datetime.timedelta(i) - datetime.timedelta(day)
        # migration = Migration_trend(test_df,lasttime, day,i)+test_df['transfer']
        #mean
        # feature_mean = test_df.groupby(['cityname', 'data']).agg({'infectnum': 'mean'})
        # feature_mean.rename(columns={'infectnum': 'infectnum_mean'}, inplace=True)
        # test_df = test_df.drop('infectnum_mean',axis=1)
        # test_df = pd.merge(test_df, feature_mean, on=['cityname', 'data'], how='left')
        # #max
        # feature_max = test_df.groupby(['cityname', 'data']).agg({'infectnum': 'max'})
        # feature_max.rename(columns={'infectnum': 'infectnum_max'}, inplace=True)
        # test_df = test_df.drop('infectnum_max',axis=1)
        # test_df = pd.merge(test_df, feature_max, on=['cityname', 'data'], how='left')
        # #sum
        # feature_sum = test_df.groupby(['cityname', 'data']).agg({'infectnum': 'sum'})
        # feature_sum.rename(columns={'infectnum': 'infectnum_sum'}, inplace=True)
        # test_df = test_df.drop('infectnum_sum', axis=1)
        # test_df = pd.merge(test_df, feature_sum, on=['cityname', 'data'], how='left')
        # #min
        # feature_min = test_df.groupby(['cityname', 'data']).agg({'infectnum': 'min'})
        # feature_min.rename(columns={'infectnum': 'infectnum_min'}, inplace=True)
        # test_df = test_df.drop('infectnum_min', axis=1)
        # test_df = pd.merge(test_df, feature_min, on=['cityname', 'data'], how='left')
        # #median
        # feature_median = test_df.groupby(['cityname', 'data']).agg({'infectnum': 'median'})
        # feature_median.rename(columns={'infectnum': 'infectnum_median'}, inplace=True)
        # test_df = test_df.drop('infectnum_median', axis=1)
        # test_df = pd.merge(test_df, feature_median, on=['cityname', 'data'], how='left')
        # #std
        # feature_std = test_df.groupby(['cityname', 'data']).agg({'infectnum': 'std'})
        # feature_std.rename(columns={'infectnum': 'infectnum_std'}, inplace=True)
        # test_df = test_df.drop('infectnum_std', axis=1)
        # test_df = pd.merge(test_df, feature_std, on=['cityname', 'data'], how='left')
        # #skew
        # feature_skew = test_df.groupby(['cityname', 'data']).agg({'infectnum': 'skew'})
        # feature_skew.rename(columns={'infectnum': 'infectnum_skew'}, inplace=True)
        # test_df = test_df.drop('infectnum_skew', axis=1)
        # test_df = pd.merge(test_df, feature_skew, on=['cityname', 'data'], how='left')
        # #kurt
        # feature_kurt = test_df.groupby(['cityname', 'data'])['infectnum'].apply(kurt).reset_index()
        # feature_kurt.rename(columns={'infectnum': 'infectnum_kurt'}, inplace=True)
        # test_df = test_df.drop('infectnum_kurt', axis=1)
        # test_df = pd.merge(test_df, feature_kurt, on=['cityname', 'data'], how='left')
        # #quantile_25
        # feature_quantile_25 = test_df.groupby(['cityname', 'data'])['infectnum'].apply(quantile_25).reset_index()
        # feature_quantile_25.rename(columns={'infectnum': 'infectnum_quantile_25'}, inplace=True)
        # test_df = test_df.drop('infectnum_quantile_25', axis=1)
        # test_df = pd.merge(test_df, feature_quantile_25, on=['cityname', 'data'], how='left')
        # #quantile_75
        # feature_quantile_75 = test_df.groupby(['cityname', 'data'])['infectnum'].apply(quantile_75).reset_index()
        # feature_quantile_75.rename(columns={'infectnum': 'infectnum_quantile_75'}, inplace=True)
        # test_df = test_df.drop('infectnum_quantile_75', axis=1)
        # test_df = pd.merge(test_df, feature_quantile_75, on=['cityname', 'data'], how='left')
        #更新
        # test_df['migration'] = migration

        # print(test_df)
        #将构建好的完整当日特征存入历史
        test_df = test_df[column_sort]
        test_df.to_csv(files, mode='a', header=False,
                        index=False,
                        sep=',')

        # 构建下一天的lagging特征
        if lag_num!=1:
            for num in range(lag_num,1,-1):
                for col_ in col_name:
                    former = col_ + '_lagging{}'.format(num)
                    latter = col_ + '_lagging{}'.format(num-1)
                    test_df[[former]] = test_df[[latter]]

        num = 1
        for col_ in col_name:
            former = col_ + '_lagging{}'.format(num)
            latter = col_
            test_df[[former]] = test_df[[latter]]

        # 将所有datea类型的日期转换为int日期格式
        data_temp = list(test_df['data'])
        for j in range(len(data_temp)):
            timeyear = str(data_temp[j].year)
            timemoth = str(data_temp[j].month)
            timeday = str(data_temp[j].day)
            time = timeyear
            if(len(timemoth)>=2):
                time = time + timemoth
            else:
                time =time + '0' + timemoth
            if (len(timeday) >= 2):
                time = time + timeday
            else:
                time = time + '0' + timeday
            data_temp[j] = time
        
        test_df['int_data'] = data_temp
        test_df[['cityname', 'region_ID', 'int_data', 'predicted']].to_csv(file, mode='a', header=False,
                                                                              index=False,
                                                                              sep=',')