import os
import sys
import argparse
import numpy as np
import pandas as pd
#from add_migration import region_migration_feature
from tqdm import tqdm
import math
import matplotlib.pyplot as plt
#@param citypath 一个城市数据文件夹地址如"./city_A"
#@param cityname 如'A'
#@return d 一个字典d["经度,纬度"] = [cityname, reginID]


#创建改时间点之前的感染人数
def  create_logging(df , df_origin, key, offset):
    df1 = df_origin.copy()
    df1['data'] =  df1['data'] + pd.DateOffset(days=offset)
    df1 = df1.rename(columns = {key: key+'_lagging' + str(offset)})
    df2 = pd.merge(df, df1[['cityname', 'region_ID','data',key+'_lagging'+ str(offset)]],
                   on= ['cityname', 'region_ID','data'],
                   how='left')
    return df2
#同比特征day代表隔几天
def WoW(df,key='infectnum',day=7):
    df_m = df.copy()
    df1 = df_m.groupby(['cityname', 'region_ID','data'])
    infectnum_list=[]
    for id_,col in tqdm(df1):
        num = col[key].values.tolist()[0]
        infectnum_list.append(num)

    # print("1111111111111111111111111")
    # print("infectnum_list:",len(infectnum_list))

    diff_list = []
    region_cnt = -1
    for nn in tqdm(range(len(infectnum_list))):
        if nn%45==0:
            region_cnt += 1
        if nn==len(infectnum_list):
            break
        if nn<=day+region_cnt*45:
            diff_num=infectnum_list[nn]
        if nn >day+region_cnt*45:
            diff_num = (infectnum_list[nn])/(infectnum_list[nn-day]+1)
            # diff_num = (infectnum_list[nn]) - infectnum_list[nn - day]

        diff_list.append(diff_num)
#    print(diff_list)
    print("diff_list",len(diff_list))
        
    return diff_list

def Sum(df,key='infectnum',day=7):
    df_m = df.copy()
    df1 = df_m.groupby(['cityname', 'region_ID','data'])

    infectnum_list=[]
    for id_,col in tqdm(df1):
#        print(col['infectnum'])
        num = col[key].values.tolist()[0]
        infectnum_list.append(num)
#    print(len(infectnum_list))

    diff_list = []
    region_cnt = -1
    for nn in tqdm(range(len(infectnum_list))):
        if nn % 45 == 0:
            region_cnt += 1
        if nn == len(infectnum_list):
            break
        if nn <= day + region_cnt * 45:
            diff_num = infectnum_list[nn]
        if nn > day + region_cnt * 45:
            diff_num = sum(infectnum_list[nn - day+1: nn+1])

        diff_list.append(diff_num)
    print(len(diff_list))
    return diff_list
def kurt(df):
    return df.kurt() 
#分位数
def quantile_25(df):
    return df.quantile(q=0.25)
def quantile_75(df):
    return df.quantile(q=0.75)
#创建infection的基本lagging数据
def create_lagging_feature(city_list, data_path, output_path,loggingNum=5,log=False):
    df = []
    #所有城市的infection.csv数据
    for i in range(len(city_list)):
        city_path = os.path.join(data_path, "city_%s"% city_list[i])
        res = pd.read_csv(os.path.join(city_path, "infection.csv"),
                         header=None, names=['cityname', 'region_ID', 'data', 'infectnum'])
        df.append(res)
    df = pd.concat(df)
    # for k in [4]:
    #     df_Wow = WoW(df,key='infectnum',day=k)#感染day天同比
    #     df['infectnum_WoW%d'%k] = df_Wow
    # for k in [4]:
    #     df_sum = Sum(df,key='infectnum',day=k)#累积day天感染人数
    #     df['infectnum_Sum%d'%k] = df_sum
    #     if log==True:
    #         df['infectnum_Sum%d' % k] = df['infectnum_Sum%d'%k].map(lambda x:math.log1p(x))

    numb = df.shape[0]
    if log == True:
        infectnum_log = []
        for nn in tqdm(range(numb)):
            infectnum_log.append(math.log(df[nn:nn + 1]['infectnum'] + 1))
        df['infectnum'] = infectnum_log

    # #    构建区域均值，求和，最大值特征
    # df_mean = df.groupby(['cityname', 'data']).agg({'infectnum': 'mean'}).reset_index()
    # df_max = df.groupby(['cityname', 'data']).agg({'infectnum': 'max'}).reset_index()
    # df_sum = df.groupby(['cityname', 'data']).agg({'infectnum': 'sum'}).reset_index()
    # df_min = df.groupby(['cityname', 'data']).agg({'infectnum': 'min'}).reset_index()
    # df_median = df.groupby(['cityname', 'data']).agg({'infectnum': 'median'}).reset_index()
    # df_std = df.groupby(['cityname', 'data']).agg({'infectnum': 'std'}).reset_index()
    # df_skew = df.groupby(['cityname', 'data']).agg({'infectnum': 'skew'}).reset_index()
    # df_kurt = df.groupby(['cityname', 'data'])['infectnum'].apply(kurt).reset_index()
    # df_quantile_25 = df.groupby(['cityname', 'data'])['infectnum'].apply(quantile_25).reset_index()
    # df_quantile_75 = df.groupby(['cityname', 'data'])['infectnum'].apply(quantile_75).reset_index()
    # ##    print(df_mean)
    # df_mean.rename(columns={'infectnum': 'infectnum_mean'}, inplace=True)
    # df = pd.merge(df, df_mean, on=['cityname', 'data'], how='left')
    # df_max.rename(columns={'infectnum': 'infectnum_max'}, inplace=True)
    # df = pd.merge(df, df_max, on=['cityname', 'data'], how='left')
    # df_sum.rename(columns={'infectnum': 'infectnum_sum'}, inplace=True)
    # df = pd.merge(df, df_sum, on=['cityname', 'data'], how='left')
    # df_min.rename(columns={'infectnum': 'infectnum_min'}, inplace=True)
    # df = pd.merge(df, df_min, on=['cityname', 'data'], how='left')
    # df_median.rename(columns={'infectnum': 'infectnum_median'}, inplace=True)
    # df = pd.merge(df, df_median, on=['cityname', 'data'], how='left')
    # df_std.rename(columns={'infectnum': 'infectnum_std'}, inplace=True)
    # df = pd.merge(df, df_std, on=['cityname', 'data'], how='left')
    # df_skew.rename(columns={'infectnum': 'infectnum_skew'}, inplace=True)
    # df = pd.merge(df, df_skew, on=['cityname', 'data'], how='left')
    # df_kurt.rename(columns={'infectnum': 'infectnum_kurt'}, inplace=True)
    # df = pd.merge(df, df_kurt, on=['cityname', 'data'], how='left')
    # df_quantile_25.rename(columns={'infectnum': 'infectnum_quantile_25'}, inplace=True)
    # df = pd.merge(df, df_quantile_25, on=['cityname', 'data'], how='left')
    # df_quantile_75.rename(columns={'infectnum': 'infectnum_quantile_75'}, inplace=True)
    # df = pd.merge(df, df_quantile_75, on=['cityname', 'data'], how='left')

#    print(df)
    #将所有int类型的日期转换为日期格式
    data_temp = list(df['data'].values)
    for i in range(len(data_temp)):
        time=list(str(data_temp[i]))
        time.insert(4,'-')
        time.insert(7,'-')
        time = ''.join(time)
        time = pd.to_datetime(time)
        data_temp[i] =time
    df['data'] = data_temp

    df1 =  create_logging(df,df,'infectnum',1)
    # df1 =  create_logging(df1,df,'infectnum_Sum4',1)
    # df1 =  create_logging(df1,df,'infectnum_WoW4',1)


    for i in range(2,loggingNum+1):
        df1 = create_logging(df1, df, 'infectnum', i)
        # df1 = create_logging(df1, df, 'infectnum_Sum4', i)
        # df1 = create_logging(df1, df, 'infectnum_WoW4', i)
#    print(df1)
    #生成day_series
    df1['day_series'] = (df1['data'] - pd.to_datetime('2120-05-01')).map(lambda x: x.days)
    # print(df1['day_series'])

    df1.to_csv(os.path.join(output_path,'day_infection_lag%d.csv'%loggingNum),index=False)
    

#根据输入的key创建对应的key_lagging
def  create_logging_auto(df , df_origin, key, offset):
    df1 = df_origin.copy()
    df1['data'] =  df1['data'] + pd.DateOffset(days=offset)
    df1 = df1.rename(columns = {key: key+'_lagging' + str(offset)})
    df2 = pd.merge(df, df1[['cityname', 'region_ID','data',key+'_lagging'+ str(offset)]],
                   on= ['cityname', 'region_ID','data'],
                   how='left')
    return df2

# create_migration_feature的调用子函数，计算推移day的migration与对应入口city的日增长的乘积
def calmigration(df, infectsum,citymigration, day):
    df_m = df.copy()
    df_m =df_m[['cityname','region_ID','data']]
    df1 = df_m.values
    df2 = []
    for date in tqdm(df1):
        date = list(date)
        #找到改天推移day日该城市的所有入口城市
        lastdata = pd.to_datetime(date[2]) - pd.DateOffset(days=day )
        m_list = citymigration.loc[(citymigration['endcity'] == date[0])&(citymigration['data'] == lastdata)]
        #有无推移时间的migration记录
        suminfect =0
        # print(date)
        if(m_list.empty == False):
            for mlast  in m_list.values:
                mlast = list(mlast)
                # print(mlast)
                cityinfect = infectsum.loc[(infectsum['cityname'] == mlast[1])&((infectsum['data'] == date[2] ))]
                # print(cityinfect)mlast[0]
                cityinfect = list(cityinfect.values[0])[2]
                suminfect = suminfect + mlast[3]*cityinfect
            date.append(suminfect)
        else:
            date.append(0)
        df2.append(date)
    df2 = pd.DataFrame(df2)
    df2.rename(columns={0: 'cityname', 1: 'region_ID', 2:'data', 3:'migration'},inplace=True)
    return df2

#添加migration特征，城市迁入和infected number的关系
def create_migration_feature(city_list,data_path, output_path, lagging):
    infectnum = []
    # 做出城市日增长汇总infectsum
    for cityname in city_list:
        temp = pd.read_csv(os.path.join(data_path,'city_%s/infection.csv' % cityname),
                           header=None, names=['cityname', 'region_ID', 'data', 'infectnum'])
        infectnum.append(temp)
    infectnum = pd.concat(infectnum, axis=0)
    infectsum = infectnum.groupby(['cityname', 'data']).agg({'infectnum': 'sum'}).reset_index()
    infectsum.rename(columns={'infectnum': 'infectdaysum'}, inplace=True)
    # 将所有int类型的日期转换为日期格式
    data_temp = list(infectsum['data'].values)
    for i in range(len(data_temp)):
        time = list(str(data_temp[i]))
        time.insert(4, '-')
        time.insert(7, '-')
        time = ''.join(time)
        time = pd.to_datetime(time)
        data_temp[i] = time
    infectsum['data'] = data_temp

    # 构建城市间移动状况dataform
    citymigration = []
    for cityname in city_list:
        temp = pd.read_csv(os.path.join(data_path,'city_%s/migration.csv' % cityname),
                           header=None, names=['data', 'startcity', 'endcity', 'index'])
        # 将所有int类型的日期转换为日期格式
        data_temp = list(temp['data'].values)
        for i in range(len(data_temp)):
            time = list(str(data_temp[i]))
            time.insert(4, '-')
            time.insert(7, '-')
            time = ''.join(time)
            time = pd.to_datetime(time)
            data_temp[i] = time
        temp['data'] = data_temp
        temp = temp.loc[temp['endcity'] == cityname]
        citymigration.append(temp)
    citymigration = pd.concat(citymigration, axis=0)
    # 在特征集中添加migration特征
    df = pd.read_csv(os.path.join(output_path, "day_infection_lag%d.csv" % lagging), delimiter=',')
    df_migration= calmigration(df, infectsum,citymigration, day=10)  # 感染day天迁移日期
    df = pd.merge(df, df_migration,
                  on=['cityname', 'region_ID', 'data'],
                  how='left')
    df['data'] = df['data'].map(lambda x: pd.to_datetime(x))
    if log == True:
        df['migration'] = df['migration'].map(lambda x: math.log1p(x))
    #将tranfer特征加入，考虑区域移动性
    df['migration'] = df['migration'] +df ['transfer']
    #对添加的migration特征做lagging
    df1 = create_logging_auto(df,df,'migration',1)
    for i in range(2, lagging + 1):
        df1 = create_logging_auto(df1, df, 'migration', i)
    df1.to_csv(os.path.join(output_path,'day_infection_lag%d.csv'%lagging),index=False)



#在特征中对应transfer.csv找到区域的入关联强度信息transfer
def create_transfer_feature(city_list,data_path, output_path, lagging):
    transfer = []
    infect = []
    # 所有城市的infection.csv数据
    for i in range(len(city_list)):
        city_path = os.path.join(data_path, "city_%s" % city_list[i])
        res = pd.read_csv(os.path.join(output_path, "%s_transfer_only.csv"%city_list[i]),
                          header=None, names=['hour','start_ID','end_ID','index'])
        ins = pd.read_csv(os.path.join(city_path, "infection.csv"),
                          header=None, names=['cityname', 'region_ID', 'data', 'infectnum'])
        transfer.append(res)
        infect.append(ins)
    transfer = pd.concat(transfer)
    infect = pd.concat(infect)
    transfer = transfer.groupby(['start_ID','end_ID']).agg({'index':'sum'}).reset_index()
    transfer = transfer.groupby('end_ID')
    result = []

    for name, data in transfer:
        # data = data.loc[data['start_ID'] != name ]
        data = data.sort_values(by=['index'], ascending=False).reset_index(drop=True)
        #排名前几的区域投入
        data = data.loc[0:10]
        # print(data)
        involve_sum = [0.0 for i in range(45)]
        for row in data.values:
            row = list(row)
            region_part = int(row[0].split('_')[1])
            city_part = row[0].split('_')[0]
            infect_part = infect.loc[(infect['cityname'] == city_part )&(infect['region_ID']  ==  region_part)]
            # print(infect_part)
            num = list(infect_part['infectnum'].values)
            involve_sum = [involve_sum[i]+ num[i]*float(row[2])*0.001  for i in range(len(involve_sum))]

        involve_sum = pd.DataFrame(involve_sum)
        involve_sum['cityname'] = name.split("_")[0]
        involve_sum['region_ID'] = int(name.split("_")[1])
        involve_sum['data']=pd.date_range('21200501',periods=45,freq='D')
        result.append(involve_sum)

    result = pd.concat(result).reset_index(drop=True)
    result.rename(columns={0: 'transfer'}, inplace=True)
    # print(result)
        # plt.plot(range(45),involve_sum)
        # infect_now = infect.loc[(infect['cityname'] == name.split("_")[0])&(infect['region_ID'] == int(name.split("_")[1]))]
        # plt.plot(range(45), infect_now['infectnum'].values,label=str(name))
        # plt.title(name)
        # plt.legend()
        # plt.show()
    df = pd.read_csv(os.path.join(output_path,'day_infection_lag%d.csv'%lagging),delimiter=',')
    df['data'] = df['data'].map(lambda x :pd.to_datetime(x))
    df1 = pd.merge(df, result,
                   on= ['cityname', 'region_ID','data'],
                   how='left')
    # print(df1)
    df1.to_csv(os.path.join(output_path,'day_infection_lag%d.csv'% lagging),index=False)

# get_grid_dict方法
#@param citypath 一个城市数据文件夹地址如"./city_A"
#@param cityname 如'A'
#@return d 一个字典d["经度,纬度"] = [cityname, reginID]
def get_grid_dict(city_path, city_name):
    d = {}
    with open(os.path.join(city_path, 'grid_attr.csv'), 'r') as f:
        for line in f:
            items = line.strip().split(',')   #分割数据[经度，纬度，ID]
            axis = ",".join(items[0:2])
            ID = items[2]
            d[axis] = "_".join([city_name, ID])

    # d = {'x,y': ID}
    return d

#coord2ID方法
#@Description 清洗transfer.csv数据，将经纬度生成对应城市的区域ID另存为"A(city)_transfer.csv"
def coord2ID(data_path, city_name, output_path):
    city_path = os.path.join(data_path, "city_%s" % city_name)
    grid_dict = get_grid_dict(city_path, city_name)  # return 字典d["经度,纬度"] = [cityname, reginID]

    trans_filename = os.path.join(city_path, "transfer.csv")
    output_file = os.path.join(output_path, "%s_transfer.csv" % (city_name))
    with open(trans_filename, 'r') as f, open(output_file, 'w') as writer:
        for line in f:
            items = line.strip().split(',')
            start_axis = ",".join(items[1:3])
            end_axis = ",".join(items[3:5])
            index = items[5]
            try:
                start_ID = grid_dict[start_axis]
                end_ID = grid_dict[end_axis]
            except KeyError: # remove no ID axis
                continue

            writer.write("%s,%s,%s,%s\n" % (items[0], start_ID, end_ID, index))

def create_density_feature(city_list, data_path, output_path, lagging):
    # density = []
    # # 所有城市的infection.csv数据
    # for i in tqdm(range(len(city_list))):
    #     df = pd.read_csv('./dataset/data_processed/%s_density.csv' % city_list[i],
    #                      header=None, names=['data', 'hour', 'city_region', 'density'])
    #     df = df.groupby(['data', 'city_region']).agg({'density': 'sum'}).reset_index()
    #     density.append(df)
    # density = pd.concat(density)
    # density = density.groupby(['city_region']).agg({'density': 'mean'}).reset_index()
    # density.to_csv(os.path.join(output_path,'temp_density.csv'),index=False)
    density = pd.read_csv('./dataset/data_processed/temp_density.csv',
                     header=None, names=['city_region', 'density'])
    city = []
    region = []
    for data in density.values:
        data = list(data)
        city.append(data[0].split('_')[0])
        region.append(int(data[0].split('_')[1]))
    density['cityname'] = city
    density['region_ID'] = region
    density.drop(['city_region'],axis=1,inplace=True)
    if log:
        density['density'] = density['density'].map(lambda x:math.log1p(x))
    df = pd.read_csv(os.path.join(output_path, 'day_infection_lag%d.csv' % lagging), delimiter=',')
    df = pd.merge(df,density,on=['cityname','region_ID'],how='left')
    df.to_csv(os.path.join(output_path, 'day_infection_lag%d.csv' % lagging), index=False)

if __name__ == "__main__":
    region_nums = [118, 30, 135, 75, 34, 331, 38, 53, 33, 8, 48]
    city_list = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K']
    lagging = 50
    log = True#是否对数据取log
    output_path= './dataset/data_processed'
    data_path = './dataset/train_data_all'
    # 创建补齐的tranfer.csv经纬度转换
    # for city_name in city_list:
    #     coord2ID(data_path, city_name, output_path)  #转换transfer
    # 创建laaging数据
    create_lagging_feature(city_list,data_path, output_path, lagging,log=log)
    # 添加density特征
    # create_density_feature(city_list, data_path, output_path, lagging)
    # 添加transfer
    # create_transfer_feature(city_list,data_path, output_path, lagging)
    #添加migration特征
    # create_migration_feature(city_list, data_path, output_path, lagging)
