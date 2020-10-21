# 2020 Baidu Big Data Competition: Top3
### 题目:
Topic-Forecasting the future incidence of highly pathogenic contagious diseases
### 团队介绍：
队名: 岳麓F4  
队员： 
* 中南大学硕士研究生：郭海富、陈宣霖、胡敏、陶泽

名次： 3/3023 , 一等奖

## 赛题描述：
针对赛题所构造的若干虚拟城市，构造传染病群体传播预测模型，根据该地区传染病的历史每日新增感染人数、城市间迁徙指数、网格人流量指数、网格联系强度和天气等数据，预测群体未来一段时间每日新增感染人数。

## 链接：
答辩PPT ： https://pan.baidu.com/s/1q9UTO6zIkMip2JVIs_eKYA  提取密码：j9fx    
方案讲解：   
竞赛故事：   

## Packages

Xgboost>=          1.0.2  
torch>=            1.0.0    
torchvision>=      0.2.2post3   
pandas>=           0.25.3   
Numpy>=            1.17.4   
python>=           3.5    

## 主要思路说明
主要分为两个模块，将预测分为:   
* lstm进行近期预测 
* Xgbosst进行长期预测

　　结合LSTM和XGBOOST模型分别进行短期和长期预测，通过分析迁移数据区分管制和非管制城市，分别对管制和非管制城市进行建模，特对需要在于lagging特征的使用，进行处理特别是结合潜伏期的环比lagging特征  
  
<img src="https://github.com/zhuwanling/2020-Baidu-Big-Data-Competition/blob/main/Image/%E5%9B%BE%E7%89%871.png" width="500" height="300"  align=center />    

###  Xgboost长期预测   
 按照城市的density文件分析人口流量管制情况（有无交通管制），对城市区域按照后续趋势进行划分城市  
 ①	后续趋于0：A，B ，C，F ，I，J，k  
 ②	后续趋于波动：，D，E，G，H  
　　针对①设计了code_78_128_55xgb模型，针对②设计了code188lag45wow8xgb模型。为微调优化线上结果，为F城市微调出code78_128的xgb模型，c城市微调出chengxun的xgb模型,GH城市微调出code188的xgb模型，以最终得到Xgboost的预测结果。微调部分模型结果对整体提升比较小。整体拓展可以使用针对①设计的code_78_128_55xgb模型，针对②设计的code188lag45wow8xgb模型为主。  
  
#### xgb特征设计：
Xgb时间序列自回归模型部分：  
　　Y(t)=a*Y(t-1)+ b*Y(t-2),其中Y(t-1)为Y在t-1时刻的值, 而 Y(t-2)为Y在t-2时刻的值, 换句话说, 现在的Y值由过去的Y值决定, 因此自变量和因变量都为自身.构建Y(t)=a*Y(t-1)+ b*Y(t-2)+.., 但并不是用的线性模型, 用的是基于树的非线性模型, 比如梯度提升树。
与时间相关的特征
1.	lagging特征: 我们的主要目的是通过前几天的infectnum来预测下一个时刻的infectnum，那么就需要构造lagging特征，lagging的个数我们取45,  也就是用lagging1, lagging2, lagging3, lagging4 和lagging5，...，lagging45来预测现在第T天的infectnum，其中lagging1表示t-1时刻的travel_time, 以此类推．通过pandas的表连接操作，我们能很容易构造出来。通过代码运行后得到的文件会在data_process文件下。
2.	基本统计特征:基本的统计特征有很多，比如infectnum的均值，方差，最大值，最小值，偏差，峰度，四分位数等特征。这些都是作为基本的统计特征。
3.	同比特征，由于疾病传播具有隐秘性和滞后性，以一周为潜伏期。因此利用第k天的数据与第k-8天的数据作差得到同比特征。

### code_78_128_55xgb：
　　通过前55天的infectnum来预测下一个时刻的infectnum，那么就需要构造lagging特征，lagging的个数我们取55,该模型针对已知人口流动靠前的进行预测。
Xgb参数：  
　　　submit_params = {  
　　　　　'learning_rate': 0.1,  
　　　　　'n_estimators': 200,  
　　　　　'subsample': 0.7,  
　　　　　'colsample_bytree': 0.85,  
　　　　　'max_depth': 7,  
　　　　　'min_child_weight': 3,  
　　　　　'reg_alpha': 2,  
　　　　　'gamma': 0.19  
　　　}  
#### code188lag45wow8xgb：
　　数据处理：我们首先对所有的数据进行log变换，这样有利于模型更快的收敛。当模型预测完后再采用取指数的方式进行数据还原成真实的预测数据，同时采用了同比特征。该模型针对人口流动限制靠后的城市进行预测。
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
微调部分:  
　　主要是根据上两版本代码进行微调参数以适应对应城市，F城市微调出code78_128的xgb模型，c城市微调出chengxun的xgb模型,GH城市微调出code188的xgb模型。
### LSTM短期预测
　　为利用多个时间序列特征，我们采用LSTM模型根据前60天的多个时间序列特征来预测未来五天的感染人数增减情况。运用infectnum, migration，transfer, density等多个时间序列特征。参考文献1进行实现。  
特征设计：  
　　['cityname', 'region_ID', 'data','infectnum', 'infectnum_mean', 'infectnum_max', 'infectnum_sum',   
　　'infectnum_min', 'infectnum_median', 'infectnum_std', 'infectnum_skew', 'infectnum_kurt',   
　　'infectnum_quantile_25', 'infectnum_quantile_75', 'migration', 'density', 'transfer']  
训练轮次：10000epoch  
最后将LSTM的预测与xgb的预测按照权重进行融合。　　


