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
pd.set_option('display.max_rows',None)
def picture_sub(path_main,subname,is_predict_diff):
    result_A=[]
    rankings_colname = ['city_id', 'region_id','date', 'predict']
    path_sub = subname
    submission = pd.read_csv(path_sub,header=None,names=['predict','city_id','region_id'])
    n = list('ABCDEFGHIJK')
#     infection_ = pd.DataFrame([])
    submission = submission.sort_values(by = ['city_id', 'region_id'], ascending=True)
    # print(submission.loc[62])
    # print(submission)
    predict = submission['predict']
    sub_real = []
    slice2 = predict.values
    slice_sub = slice2.tolist()
    len_flag=[0]
    for name in n:
        path = path_main+'/infection%s.csv'%name
        rankings_colname = ['city_id', 'region_id','date', 'infectnum']
        infection = pd.read_csv(path,header=None,names = rankings_colname)
        infection = infection.sort_values(by = ['city_id', 'region_id'], ascending=True)
        # print(infection)
        infectnum = infection['infectnum']
        slice1 = infectnum.values
        slice_ = slice1.tolist()
        #length prediect
        predict_len =5
        for i in range(len(slice_)//60):
            # print(i)
            index_s = i*60
            index_e = (i+1)*60
            index_s_sub = i*predict_len+sum(len_flag)
            index_e_sub = (i+1)*predict_len+sum(len_flag)
            flag = (i + 1) * predict_len
            # print(index_e_sub - index_s_sub)
            data = slice_[index_s:index_e]
            data_sub = slice_sub[index_s_sub:index_e_sub]
            # print(data_sub)
            if(is_predict_diff):
                for j in range(len(data_sub)):
                    if j == 0:
                        data_sub[j] = data[len(data)-1] + float(data_sub[j])
                        if (data_sub[j] < 0):
                            data_sub[j] = 0
                    else:
                        data_sub[j] = float(data_sub[j]) + data_sub[j-1]
                        if(data_sub[j]<0):
                            data_sub[j] = 0
                sub_real.append(data_sub)
            # print(data_sub)
            merge_ = data+data_sub
            plt.plot(merge_,'g-',label='dwell')
        len_flag.append(flag)
        plt.title(name)
        # plt.show()
    return sub_real


def getdiff(stander,df_our):
    rankings_colname = ['city_id', 'region_id', 'date', 'infected']
    df_stander = pd.read_csv(stander, header=None, names=rankings_colname)
    df_resolute = df_our
    diff = pd.merge(df_stander,df_resolute,on=['city_id','region_id', 'date'],how='left')
    rmsle =  math.sqrt(mean_squared_log_error(diff['infected_x'], diff['infected_y']))
    print("与1.14提交之间的rmsle",rmsle)



def submission(sub_real,stander):
    # print(len(sub_real))
    rankings_colname = ['city_id', 'region_id', 'date', 'infected']
    df_stander = pd.read_csv(stander, header=None, names=rankings_colname)
    df_stander = df_stander.sort_values(by=['city_id', 'region_id'], ascending=True).reset_index(drop=True)
    # print(df_stander)
    df1 = df_stander.groupby(['city_id','region_id'])
    infectnum = []
    index = 0
    predict_len = 5
    for name,data in df1 :
        # print(name)
        # print(data)
        if index>=392 and index<723:            #除开A区域的神经网络预测
            infect = list(data['infected'].values[predict_len:])
            avg_sub = []
            for  i in range(predict_len):
                avg_sub.append((sub_real[index][i]*0.01+list(data['infected'].values[0:predict_len])[i]*0.99))
            temp = avg_sub+infect
        # print(temp)
        else:
            temp = list(data['infected'])
        # print(temp)
        infectnum.append(pd.DataFrame(temp))
        index+=1

    infectnum = pd.concat(infectnum)
    infectnum.reset_index(drop=True,inplace=True)
    df_stander['infected'] = infectnum
    # print(df_stander)
    return df_stander
    # print(df_stander)

# if __name__ == "__main__":
is_predict_diff = True
def run():
    resolute = './submission/resolute.csv'
    stander = './submission/result.csv'
    sub_real = picture_sub('./lstm/data/infection/',resolute,is_predict_diff)
    if is_predict_diff:
        df_our = submission(sub_real,stander)
        df_our.to_csv('./submission/nearfuture.csv',index=False,header=None)
        getdiff(stander,df_our)

