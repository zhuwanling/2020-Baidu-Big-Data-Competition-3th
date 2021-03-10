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
# pd.set_option('display.max_row',None)
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


def getdiff(stander,resolute1,resolute2,resolute3,resolute4,resolute5):
    rankings_colname = ['city_id', 'region_id', 'date', 'infected']
    df_stander = pd.read_csv(stander, header=None, names=rankings_colname)
    #code188lag45wow8
    df_resolute1 = pd.read_csv(resolute1, header=None, names=rankings_colname)
    #code78_128
    df_resolute2 = pd.read_csv(resolute2, header=None, names=rankings_colname)
    #chengxun
    df_resolute3 = pd.read_csv(resolute3, header=None, names=rankings_colname)
    #code_78_128_55
    df_resolute4 = pd.read_csv(resolute4, header=None, names=rankings_colname)
    #code188
    df_resolute5 = pd.read_csv(resolute5, header=None, names=rankings_colname)

    #平均result5与result1
    df_avg = [df_resolute5, df_resolute1]
    df_avg = pd.concat(df_avg).reset_index()
    df_avg = df_avg.groupby(['city_id', 'region_id', 'date']).agg({'infected': 'mean'}).reset_index()

    df_resolute1_choose = ['D','E' ]
    df_resolute2_choose = ['F']
    df_resolute3_choose = ['C' ]
    df_resolute4_choose = ['A', 'B','I', 'J','K']
    #df_resolute5_choose = ['H']
    df_avg_choose  = ['G','H']
    fuse_resolute =[
                    df_resolute1[df_resolute1.city_id.isin(df_resolute1_choose)].reset_index(drop=True),
                    df_resolute2[df_resolute2.city_id.isin(df_resolute2_choose)].reset_index(drop=True),
                    df_resolute3[df_resolute3.city_id.isin(df_resolute3_choose)].reset_index(drop=True),
                    df_resolute4[df_resolute4.city_id.isin(df_resolute4_choose)].reset_index(drop=True),
                    # df_resolute5[df_resolute5.city_id.isin(df_resolute5_choose)].reset_index(drop=True),
                    df_avg[df_avg.city_id.isin(df_avg_choose)].reset_index(drop=True),
                    ]
    fuse_resolute = pd.concat(fuse_resolute).reset_index(drop=True)
    # print(fuse_resolute)
    fuse_resolute.to_csv('./submission/result.csv', index=False, header=None)
    diff = pd.merge(df_stander,fuse_resolute,on=['city_id','region_id', 'date'],how='left')
    # print(diff)
    rmsle =  math.sqrt(mean_squared_log_error(diff['infected_x'], diff['infected_y']))
    rmse = mean_squared_log_error(diff['infected_x'], diff['infected_y'])
    print("与参考提交之间的rmsle",rmsle)
    print("与参考提交之间的rmse", rmse)



# if __name__ == "__main__":
    is_predict_diff = False
    is_fuse = True
def fusion():
    stander = './submission/+0.94324_nearfuture.csv'
    resolute1 = './code188lag45wow8/submission/day000_submission_45_03_.csv'
    resolute2 = './code78_128/submission/day000_submission_50_01.csv'
    resolute3 = './chengxun/submission/chegnxuan_45_01.csv'
    resolute4 = './code_78_128_55/submission/day000_submission_55_01.csv'
    resolute5 = './code_188(1)/submission/day000_submission_50_03_.csv'
    # picture_sub('./dataset/train_data_all/',resolute,is_predict_diff)
    getdiff(stander,resolute1,resolute2,resolute3,resolute4,resolute5)

