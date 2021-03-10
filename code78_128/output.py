##画图
'''
submission 请放在train_data文件夹下；如果是Linux系统，请把\\改为/即可
输入为:'主路径' 和 'submission.csv'(默认名称)
如下文件夹树形式
'''
import pandas as pd
import matplotlib.pyplot as plt
import math
from sklearn.metrics import  mean_squared_log_error
import  os
pd.set_option('display.max_row',None)
def picture_sub(path_main,subname,is_predict_diff):
    result_A=[]
    rankings_colname = ['city_id', 'region_id','date', 'predict']
    path_sub = subname
    submission = pd.read_csv(path_sub,header=None,names=rankings_colname)
    n = list('ABCDEFGHIJK')
    # n = ['B','D','E','F','G','H']
#     infection_ = pd.DataFrame([])
    submission = submission.sort_values(by = ['city_id', 'region_id'], ascending=True)
    # print(submission.loc[62])
    # print(submission)
    predict = submission['predict']
    slice2 = predict.values
    slice_sub = slice2.tolist()
    len_flag=[0]
    for j in n:
        name = 'city_'+j
        path = path_main+'{}\\infection.csv'.format(name)
        rankings_colname = ['city_id', 'region_id','date', 'infectnum']
        infection = pd.read_csv(path,header=None,names = rankings_colname)
        infection = infection.sort_values(by = ['city_id', 'region_id'], ascending=True)
        infectnum = infection['infectnum']
        slice1 = infectnum.values
        slice_ = slice1.tolist()

        for i in range(len(slice_)//60):
            index_s = i*60
            index_e = (i+1)*60
            index_s_sub = i*30+sum(len_flag)
            index_e_sub = (i+1)*30+sum(len_flag)
            flag = (i+1)*30
            data = slice_[index_s:index_e]
            data_sub = slice_sub[index_s_sub:index_e_sub]
            if(is_predict_diff):
                for j in range(len(data_sub)):
                    if j == 0:
                        data_sub[j] = data[len(data)-1] + data_sub[j]
                    else:
                        data_sub[j] = data_sub[j] + data_sub[j-1]
            merge_ = data+data_sub
            plt.plot(merge_,'g-',label='dwell')
        len_flag.append(flag)
        plt.title(name)
        plt.show()


def getdiff(stander,resolute):
    rankings_colname = ['city_id', 'region_id', 'date', 'infected']
    df_stander = pd.read_csv(stander, header=None, names=rankings_colname)
    df_resolute = pd.read_csv(resolute, header=None, names=rankings_colname)
    #对两个数据进行城市融合
    if is_fuse:
        df_stander_choose = ['A', 'B','C','D','E','F','G','H', 'I', 'J','K' ]
        df_resolute_choose = []
        # print(df_stander[df_stander.city_id.isin(df_stander_choose)].reset_index(drop=True))
        # print(df_resolute[df_resolute.city_id.isin(df_resolute_choose)].reset_index(drop=True))
        fuse_resolute =[df_stander[df_stander.city_id.isin(df_stander_choose)].reset_index(drop=True),df_resolute[df_resolute.city_id.isin(df_resolute_choose)].reset_index(drop=True)]
        fuse_resolute = pd.concat(fuse_resolute).reset_index(drop=True)
        # fuse_resolute.to_csv('./submission/temp.csv', index=False, header=None)
    else:
        fuse_resolute = df_resolute
    diff = pd.merge(df_stander,fuse_resolute,on=['city_id','region_id', 'date'],how='left')
    # print(diff)
    rmsle =  math.sqrt(mean_squared_log_error(diff['infected_x'], diff['infected_y']))
    rmse = mean_squared_log_error(diff['infected_x'], diff['infected_y'])
    print("与参考提交之间的rmsle",rmsle)
    print("与参考提交之间的rmse", rmse)



if __name__ == "__main__":
    is_predict_diff = False
    is_fuse = False
    stander = './submission/50_onlylagging_1.28.csv'
    resolute = './submission/day000_submission_50_01.csv'
    picture_sub('./dataset/train_data_all/',resolute,is_predict_diff)
    getdiff(stander,resolute)

