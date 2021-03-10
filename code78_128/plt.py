##画图
'''
submission 请放在train_data文件夹下；如果是Linux系统，请把\\改为/即可
输入为:'主路径' 和 'submission.csv'(默认名称)
如下文件夹树形式
train_data
    CityA
        infection.csv
    CityB
        infection.csv
    CityC
        infection.csv
    CityD
        infection.csv
    CityE
        infection.csv
    submission.csv
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
def picture_sub(path_main,subname='submission.csv'):
    result_A=[]
    rankings_colname = ['city_id', 'region_id','date', 'number_of_newly_infected_persons']
    path_sub = path_main+subname
    print(path_sub)
    submission = pd.read_csv(path_sub,header=None,names=rankings_colname)
    n = list('ABCDE')
#     infection_ = pd.DataFrame([])
    submission = submission.sort_values(by = ['city_id', 'region_id'], ascending=True)
    # print(submission.loc[62])
    # print(submission)
    number_of_newly_infected_persons_sub = submission['number_of_newly_infected_persons']
    slice2 = number_of_newly_infected_persons_sub.values
    slice_sub = slice2.tolist()
    slice_sub_ = slice2.tolist()
#    slice_sub=[]
#    for nn in slice_sub_:
#        num = math.exp(nn)-1
#        if num<0:
#            num=0
#        num = int(num)
#        slice_sub.append(num)
    len_flag=[0]
    for j in n:
        name = 'city_'+j
        print(name)
        path_main
        path = path_main+'{}\\infection.csv'.format(name)
    #     print(path)
        rankings_colname = ['city_id', 'region_id','date', 'number_of_newly_infected_persons']
        infection = pd.read_csv(path,header=None,names = rankings_colname)
        infection = infection.sort_values(by = ['city_id', 'region_id'], ascending=True)
#         infection_ = pd.concat([infection_,infection],axis=0)    

        number_of_newly_infected_persons = infection['number_of_newly_infected_persons']   
        slice1 = number_of_newly_infected_persons.values
        slice_ = slice1.tolist()
        print(len(slice_))

        for i in range(len(slice_)//45):
            index_s = i*45
            index_e = (i+1)*45

            index_s_sub = i*30+sum(len_flag)
            index_e_sub = (i+1)*30+sum(len_flag)
            flag = (i+1)*30
            data = slice_[index_s:index_e]
            data_sub = slice_sub[index_s_sub:index_e_sub]
            merge_ = data+data_sub 
    #         print(len(merge_))
            plt.plot(merge_,'g-',label='dwell')
        len_flag.append(flag)
    #     print(len_flag)
    #     print(index_e_sub)
        plt.show()

picture_sub('.\\train_data\\','day000_submission_30_01.csv')