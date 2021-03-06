import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import  mean_squared_log_error
from .ultis_1 import *
import  pandas as pd
import math

pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
def evalerror(preds, dtrain):
    labels = dtrain.get_label()
#    if preds==0:
#        preds
    preds[preds<0]=0
    return 'rmsle', math.sqrt(mean_squared_log_error(preds,labels))

def xgboost_submit(df, params,lag):
    # train_df = df.loc[df['data'] < pd.to_datetime('2120-06-15')]
    # train_df = df[df.cityname.isin(['B','D','E','F','G','H'])]
    # train_df = df.dropna()
    train_df = df
    X = train_df[train_feature].values
    y = train_df['infectnum'].values
#    X_train, y_train = X,y
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01, random_state=0)

    eval_set = [(X_test, y_test)]
    regressor = xgb.XGBRegressor(learning_rate=params['learning_rate'], n_estimators=params['n_estimators'],
                                 booster='gbtree', objective='reg:linear', n_jobs=-1, subsample=params['subsample'],
                                 colsample_bytree=params['colsample_bytree'], random_state=0,
                                 max_depth=params['max_depth'], gamma=params['gamma'],
                                 min_child_weight=params['min_child_weight'], reg_alpha=params['reg_alpha'])
    # regressor.fit(X_train, y_train, verbose=True)
    regressor.fit(X_train, y_train, verbose=True, early_stopping_rounds=10, eval_metric=evalerror,
                  eval_set=eval_set)
    feature_vis(regressor, train_feature)
    joblib.dump(regressor, './code188lag45wow8/model/xgbr.pkl')
#    print(regressor)
    submission(train_feature, regressor, train_df, './code188lag45wow8/submission/day000_submission_{}_03_.csv'.format(lag),lag,log)

def fit_evaluate(df, df_test, params):
    df = df.dropna()    #删除所有有NAN的行
    X = df[train_feature].values
    y = df['infectnum'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01, random_state=0)

    df_test = df_test[valid_feature].values
    valid_data = bucket_data(df_test)
    eval_set = [(X_test, y_test)]
    regressor = xgb.XGBRegressor(learning_rate=params['learning_rate'], n_estimators=params['n_estimators'],
                                 booster='gbtree', objective='reg:linear', n_jobs=-1, subsample=params['subsample'],
                                 colsample_bytree=params['colsample_bytree'], random_state=0,
                                 max_depth=params['max_depth'], gamma=params['gamma'],
                                 min_child_weight=params['min_child_weight'], reg_alpha=params['reg_alpha'])
    regressor.fit(X_train, y_train, verbose=False, early_stopping_rounds=10, eval_metric=evalerror,
                  eval_set=eval_set)
    # feature_vis(regressor, train_feature) 

    return regressor, cross_valid(regressor, valid_data,
                                  lagging=lagging), regressor.best_iteration, regressor.best_score


def train(df, params, best, vis=False):
    train1 = df.loc[df['data'] <= pd.to_datetime('2120-05-09')]
    train2 = df.loc[
        (df['data'] > pd.to_datetime('2120-05-09')) & (
            df['data'] <= pd.to_datetime('2120-05-18'))]
    train3 = df.loc[
        (df['data'] > pd.to_datetime('2120-05-18')) & (
            df['data'] <= pd.to_datetime('2120-05-27'))]
    train4 = df.loc[
        (df['data'] > pd.to_datetime('2120-05-27')) & (
            df['data'] <= pd.to_datetime('2120-06-05'))]
    train5 = df.loc[
        (df['data'] > pd.to_datetime('2120-06-05')) & (
            df['data'] <= pd.to_datetime('2120-06-14'))]

    regressor, loss1, best_iteration1, best_score1 = fit_evaluate(pd.concat([train1, train2, train3, train4]), train5,
                                                                  params)
    print (best_iteration1, best_score1, loss1)

    regressor, loss2, best_iteration2, best_score2 = fit_evaluate(pd.concat([train1, train2, train3, train5]), train4,
                                                                  params)
    print (best_iteration2, best_score2, loss2)

    regressor, loss3, best_iteration3, best_score3 = fit_evaluate(pd.concat([train1, train2, train4, train5]), train3,
                                                                  params)
    print (best_iteration3, best_score3, loss3)

    regressor, loss4, best_iteration4, best_score4 = fit_evaluate(pd.concat([train1, train3, train4, train5]), train2,
                                                                  params)
    print (best_iteration4, best_score4, loss4)

    regressor, loss5, best_iteration5, best_score5 = fit_evaluate(pd.concat([train2, train3, train4, train5]), train1,
                                                                  params)
    print (best_iteration5, best_score5, loss5)

    if vis:
        xgb.plot_tree(regressor, num_trees=5)
        results = regressor.evals_result()
        epochs = len(results['validation_0']['rmse'])
        x_axis = range(0, epochs)
        fig, ax = plt.subplots()
        ax.plot(x_axis, results['validation_0']['rmse'], label='Train')
        ax.plot(x_axis, results['validation_1']['rmse'], label='Test')
        ax.legend()
        plt.ylabel('rmse Loss')
        plt.ylim((0.2, 0.3))
        plt.show()

    loss = [loss1, loss2, loss3, loss4, loss5]
    params['loss_std'] = np.std(loss)
    params['loss'] = str(loss)
    params['mean_loss'] = np.mean(loss)
    params['n_estimators'] = str([best_iteration1, best_iteration2, best_iteration3, best_iteration4, best_iteration5])
    params['best_score'] = str([best_score1, best_score2, best_score3, best_score4, best_score5])

    print(str(params))
    if np.mean(loss) <= best:
        best = np.mean(loss)
        print("best with: " + str(params))
        feature_vis(regressor, train_feature)
    return best



# if __name__ == "__main__":
lagging = 45
log=True#是否已经对数据取log
# df = pd.read_csv('./dataset/data_processed/infection_lag5.csv',header=None, parse_dates =['data'],
#                  names=['cityname', 'region_ID','data','infectnum','lagging1',
#                         'lagging2','lagging3','lagging4','lagging5'])
df = pd.read_csv('./code188lag45wow8/dataset/data_processed/day_infection_lag{}.csv'.format(lagging),delimiter=',', parse_dates =['data'])
feature = df.columns.values.tolist()
#    print(feature)
#    lagging_feature = ['lagging%01d' % e for e in range(lagging, 0, -1)]
lagging_feature=[x for x in df.columns.values.tolist() if x not in ['cityname', 'region_ID', 'data', 'infectnum',
                     'infectnum_Sum', 'infectnum_mean', 'infectnum_max', 'infectnum_sum',
                    'infectnum_median','infectnum_std','infectnum_skew','migration','transfer','density','infectnum_min',
                    'infectnum_kurt','infectnum_quantile_25','infectnum_quantile_75','day_series','infectnum_WoW7','infectnum_Sum3','infectnum_WoW3','infectnum_Sum3']]
                    #'infectnum_mean', 'infectnum_max', 'infectnum_sum','infectnum_min','infectnum_WoW',
                    # 'infectnum_median','infectnum_std','infectnum_skew','day_series','migration',
                    # 'infectnum_kurt','infectnum_quantile_25','infectnum_quantile_75', 'infectnum_WoW', 'infectnum_Sum',
base_feature = [x for x in df.columns.values.tolist() if x not in ['cityname', 'region_ID', 'data', 'infectnum',
                    'migration','day_series']]
base_feature = [x for x in base_feature if x not in lagging_feature]
train_feature = list(base_feature)

train_feature.extend(lagging_feature)
#    print(lagging_feature)
#    print(0/0)
valid_feature = list(base_feature)
valid_feature.extend(['day_series', 'infectnum'])


def run():
    print("训练集特征：",train_feature)
    print("验证集特征：",valid_feature)


    # ----------------------------------------Train-------------------------------------------
#    params_grid = {
#         'learning_rate': [0.05],
#         'n_estimators': [100],
#         'subsample': [0.7],
#         'colsample_bytree': [0.7],
#         'max_depth': [7],
#         'min_child_weight': [3],
#         'reg_alpha': [2],
#         'gamma': [0.1]
#     }
#    #
#    # grid = ParameterGrid(params_grid)
#    # best = 1
#    #
#    # for params in grid:
#    #     best = train(df, params, best)
#    grid = ParameterGrid(params_grid)
#    best = 0
#
#    for params in grid:
#        best = train(df, params, best)
    # ----------------------------------------submission-------------------------------------------
#    submit_params = {
#        'learning_rate': 0.1,
#        'n_estimators': 300,
#        'subsample': 0.7,#0.7
#        'colsample_bytree': 0.95,#0.85
#        'max_depth': 8,#7
#        'min_child_weight': 4,#3
#        'reg_alpha': 2.5,#2
#        'gamma': 0.19
#    }
#    1.88-1.22交换->1.09
    submit_params = {
        'learning_rate': 0.1,
        'n_estimators': 200,
        'subsample': 0.9,#0.7
        'colsample_bytree': 0.45,#0.85
        'max_depth': 9,#7
        'min_child_weight': 3,#3
        'reg_alpha': 1,#2
        'gamma': 0.19
    }
#    submit_params = {
#        'learning_rate': 0.1,
#        'n_estimators': 200,
#        'subsample': 0.9,#0.7
#        'colsample_bytree': 0.45,#0.85
#        'max_depth': 9,#7
#        'min_child_weight': 3,#3
#        'reg_alpha': 1,#2
#        'gamma': 0.25#0.19
#    }
        
    xgboost_submit(df, submit_params,lag=lagging)
    if log==True:
        sub = pd.read_csv('./code188lag45wow8/submission/day000_submission_{}_03_.csv'.format(lagging),header=None, names=['cityname', 'region_ID', 'data', 'infectnum'])
        list_sub = sub['infectnum'].values
        list_sub_m = list_sub.tolist()
        list_sub_f = []
        for nn in list_sub_m:
            num = math.exp(nn)-1
            if num<0:
                num = 0
            list_sub_f.append(num)
        sub['infectnum']=list_sub_f
        sub.to_csv('./code188lag45wow8/submission/day000_submission_{}_03_.csv'.format(lagging),header=False,index=False)
        
