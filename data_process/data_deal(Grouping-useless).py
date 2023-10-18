# coding: utf-8
'''
  这里对提取出来的数据，进行预处理：
       *均值或者线性插值填充，根据数据画图情况进行填充
       删了各期尿量, 'Hct-pre.1'

'''
from copy import deepcopy
import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import StandardScaler
import missingno as msno
from matplotlib import pyplot as plt
#from sklearn.preprocessing import Imputer

import csv
from sklearn.ensemble import RandomForestRegressor



plt.rc("font",family='KaiTi')

# ,index_col=0
medical_data = pd.read_csv('./data/features.csv')
medical_labels = pd.read_csv('./data/labels.csv')
#print(medical_data)

def show_missing(data):
  print('____________查看缺失值缺失情况___________________')
  # msno.bar(medical_data.sample(medical_data.shape[0]-1,), figsize=(12, 7), )
  #调试
  #真的做了采样吗...
  # medical_data_sample=data.sample(data.shape[0]-1,)#抽取样本, #data.shape[0]=425
  # print('shape of medical_data_sample: ',medical_data_sample.shape,'\ntype of medical_data_sample: ',type(medical_data_sample))
  # i=0
  # for index,colum in medical_data_sample.items():
  # 	if i<3:
  # 		i+=1
  # 		print(index,':',colum)
  # msno.bar(medical_data_sample,labels=medical_data_sample.columns.tolist(), figsize=(12, 7))
  msno.bar(data,labels=data.columns.tolist(), figsize=(12, 7))
  #调试end


  plt.show()


def missing_values_table(df):
  mis_val = df.isnull().sum()  # 总缺失值
  mis_val_percent = 100 * df.isnull().sum() / len(df)  # 缺失值比例
  # mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)  # 缺失值制成表格
  # mis_val_table_ren_columns = mis_val_table.rename(columns={0: 'Missing Values',
  #                                                           1: '% of Total Values'})
  # 将index单独存成一列feature name  
  mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)  # 缺失值制成表格
  mis_val_table.insert(0, 'feature name', mis_val.index)
  mis_val_table_ren_columns = mis_val_table.rename(columns={0: 'Missing Values',
                                                            1: '% of Total Values'})
  # # 缺失值比例列由大到小排序
  # mis_val_table_ren_columns = mis_val_table_ren_columns[
  # mis_val_table_ren_columns.iloc[:, 2] != 0].sort_values('% of Total Values', ascending=False).round(1)

  print('Your selected dataframe has {} columns.\nThere are {} columns that have missing values.'.format(df.shape[1],
                                                                        mis_val_table_ren_columns.shape[0]))
  mis_val_table_ren_columns.to_csv('./data/mis_val.csv',index=False)

  return mis_val_table_ren_columns

# missing_values_table(medical_data)

# 获取缺失值比例 < 60% 的列
def get_remaining_features(df):
  missing_df = pd.read_csv('./data/mis_val.csv')
  # missing_columns = list(missing_df[missing_df['% of Total Values'] > 60].index)
  # print(missing_df['% of Total Values'] > 60)#index bool
  # missing_columns_name=missing_columns['feature name']
  missing_columns = missing_df[missing_df['% of Total Values'] > 60]['feature name'].tolist()
  print('We will remove %d columns.' % len(missing_columns))
  #print(missing_columns)

  # 删除缺失值比例高于60%的列
  # df = df.drop(columns = list(missing_columns))
  df = df.drop(columns = missing_columns)
  # print("remaining features:\n",medical_data)
  df.to_csv('./data/remaining_features.csv',index=False)
  return df

# 获取缺失值比例 <= 60% >0%的列
def get_need_dealing_features(df):
  missing_df = pd.read_csv('./data/mis_val.csv')
  # missing_columns = list(missing_df[(missing_df['% of Total Values'] <= 60) and (missing_df['% of Total Values'] > 0)].index)
  missing_columns = missing_df[(missing_df['% of Total Values'] <= 60) & (missing_df['% of Total Values'] > 0)]['feature name'].tolist()
  print('We will get %d columns.' % len(missing_columns))
  #print(missing_columns)

  # 选出需要处理的列
  # df = df.drop(columns = list(missing_columns))
  # df = df[list(missing_columns)]
  df = df[missing_columns]

  # print("remaining features:\n",medical_data)
  df.to_csv('./data/need_dealing_features.csv',index=False)
  return df

def predict_with_sum(tabel, pre_x_names, pre_y_name):
  # 筛选出y含空值的, 然后取x的列
  data_isnull=tabel.loc[tabel[pre_y_name].isnull()]
  X=data_isnull[pre_x_names]
  # 求和得Y
  Y = X.sum(axis=1) # 每行求和
  print("(X_pre):\n{}\n{}(Y_pre):\n{}".format(X,pre_y_name,Y))
  return Y

def predict_with_1_label(tabel, pre_x_name, pre_y_name):
  data_notnull = tabel.loc[tabel[pre_y_name].notnull()]
  data_isnull = tabel.loc[tabel[pre_y_name].isnull()].drop( [pre_y_name],axis = 1)
  data_notnull_filled=data_notnull.fillna(tabel.median())
  #  data_isnull_filled=data_isnull.fillna(tabel1.median().drop( [pre_y_name]))
  data_isnull_filled=data_isnull[[pre_x_name]].fillna(tabel.median()[pre_x_name])
  # 随机森林填充
  #  X = data_notnull_filled.drop([pre_y_name],axis = 1)
  X = np.array(data_notnull_filled[pre_x_name]).reshape(-1, 1)
  Y =data_notnull_filled[pre_y_name]
  rfr = RandomForestRegressor(n_estimators=100, n_jobs=-1)
  rfr.fit(X, Y)
  predict_null = rfr.predict(np.array(data_isnull_filled[pre_x_name]).reshape(-1, 1))
  print("data_isnull_filled(X_pre):\n{}\n{}(Y_pre):\n{}".format(data_isnull_filled,pre_y_name,predict_null))
  return predict_null

def predict_with_labels_dropOtherNA(tabel, pre_x_names, pre_y_name):
  features=pre_x_names.copy()
  features.append(pre_y_name)
  # print("dropOtherNA tabel:\n",tabel)
  data=tabel[features].copy()
  data_notnull = data.loc[data[pre_y_name].notnull()]
  data_isnull = data.loc[data[pre_y_name].isnull()].drop( [pre_y_name],axis = 1)
  other_NA=data_notnull[data_notnull.isnull().values==True]
  data_notnull_filled=data_notnull.drop(other_NA.index)#挑选其他特征也没有缺失的用来训练模型
  # print("other_NA:\n",other_NA)
  #  data_isnull_filled=data_isnull.fillna(tabel1.median().drop( [pre_y_name]))
  data_isnull_filled=data_isnull[pre_x_names].fillna(data.median()[pre_x_names])#中位数填充
  # 随机森林填充
  X = data_notnull_filled.drop([pre_y_name],axis = 1)
  Y =data_notnull_filled[pre_y_name]
  # print("X:\n",X[X.isnull().values==True])
  rfr = RandomForestRegressor(n_estimators=100, n_jobs=-1)
  rfr.fit(X, Y)
  # predict_null = rfr.predict(np.array(data_isnull_filled[pre_x_name]).reshape(-1, 1))
  predict_null = rfr.predict(data_isnull_filled[pre_x_names])
  print("data_isnull_filled(X_pre):\n{}\n{}(Y_pre):\n{}".format(data_isnull_filled,pre_y_name,predict_null))
  return predict_null

def fill_tabel1(all_data,features,labels):
  print('-----------------tabel1一般情况------------------')
  #  随机森林辅助中位数填补
  tabel1=all_data[features]
  tabel1_new=tabel1.copy()
  pre_y_names=['身高','体重','BMI']
  for pre_y_name in pre_y_names:
    predict_null=predict_with_1_label(tabel1, '年龄', pre_y_name)
    tabel1_new.loc[tabel1[pre_y_name].isnull(), pre_y_name] = predict_null
  #  保存填充后的值
  all_data.loc[:,features] = tabel1_new
  #  乙肝携带所有可能的值填充（Assigning All Possible values of the Attribute）
  data_isnull = all_data.loc[all_data['乙肝携带'].isnull()]
  data_isnull_idx=all_data.loc[all_data['乙肝携带'].isnull()].index#index既要用来删除data又要用来增删label
  data_isnull_filled0=data_isnull.fillna({'乙肝携带':0})
  data_isnull_filled1=data_isnull.fillna({'乙肝携带':1})
  all_data=all_data.drop(index=data_isnull_idx)  
  all_data = pd.concat([all_data, data_isnull_filled0, data_isnull_filled1], axis=0, sort=False)
  labels_filled=labels.loc[data_isnull_idx]
  labels = labels.drop(index=data_isnull_idx)
  labels = pd.concat([labels, labels_filled, labels_filled], axis=0, sort=False)
  all_data = all_data.reset_index(drop=True)
  labels = labels.reset_index(drop=True)
  # print("data_index:\n{}\nlabel_index:\n{}".format(all_data.index,labels.index))
  return all_data,labels

def fill_tabel2(all_data,features):
  print('-----------------tabel2术中情况------------------')
  tabel=all_data[features]
  tabel_new=tabel.copy()
  # 术式(经典1背驮2)处理成术式(经典1背驮0)
  tabel_new.loc[:,'术式(经典1背驮2)']  = tabel.loc[:,'术式(经典1背驮2)'].replace(2,0)
  tabel=tabel_new.copy()

  #  随机森林填补手术时间/无肝期时间min
  # 利用术式(经典1背驮2)和无肝期时间拟合手术时间
  pre_y_names=['手术时间min','无肝期时间min']
  for idx,pre_y_name in enumerate(pre_y_names):
    predict_null=predict_with_labels_dropOtherNA(tabel, ['术式(经典1背驮2)',pre_y_names[(idx+1)%2]], pre_y_name)#用一个做X的时候就用另一个做y
    tabel_new.loc[tabel[pre_y_name].isnull(), pre_y_name] = predict_null
  tabel=tabel_new.copy()

  #填充冷热缺血时间
  # null_index=tabel[tabel['热缺血时间min'].isnull().values==True].index
  tabel_new=tabel.fillna({'热缺血时间min':tabel.median()['热缺血时间min'], '冷缺血时间min':tabel.median()['冷缺血时间min']})
  tabel=tabel_new.copy()

  # 总入量(可以根据注射液体求和精确计算)注意此处若白蛋白等液体有缺失样例, 应在白蛋白之后填充
  pre_x_names=['红细胞','血浆','自体血','4%白蛋白','2%白蛋白','纯白蛋白g','NS','LR','万汶','佳乐施']
  pre_y_name='总入量'
  tabel_new.loc[tabel[pre_y_name].isnull(), pre_y_name]=predict_with_sum(tabel, pre_x_names, pre_y_name)
  tabel=tabel_new.copy()

  # 填充白蛋白,出血量
  pre_x_names=['红细胞','血浆','自体血','2%白蛋白','纯白蛋白g','NS','LR','万汶','佳乐施','总入量']
  pre_y_names=['4%白蛋白','出血量']
  for idx,pre_y_name in enumerate(pre_y_names):
    add_x_temp=pre_y_names.copy()
    x_temp=pre_x_names.copy()
    add_x_temp.remove(pre_y_name)
    x_temp.extend(add_x_temp)
    # print("x_temp:\n",x_temp)
    predict_null=predict_with_labels_dropOtherNA(tabel, x_temp, pre_y_name)#用一个做X的时候就用另一个做y
    tabel_new.loc[tabel[pre_y_name].isnull(), pre_y_name] = predict_null
  tabel=tabel_new.copy()

  # 填充Ⅲ期尿量
  pre_x_names=['I期尿量','II期尿量','胸水','腹水','速尿mg','甘露醇ml','碳酸氢钠ml','总入量','出血量']
  pre_y_name='III期尿量'
  predict_null=predict_with_labels_dropOtherNA(tabel, pre_x_names, pre_y_name)
  tabel_new.loc[tabel[pre_y_name].isnull(), pre_y_name] = predict_null
  tabel=tabel_new.copy()
  
  # 填充总尿量(缺失特征少的单个样本)
  pre_x_names=['I期尿量','II期尿量','III期尿量']
  pre_y_name='总尿量'
  # 找到'总尿量'为空'I期尿量'的样本
  tabel_new.loc[(tabel[pre_y_name].isnull() & tabel['I期尿量'].notnull()), pre_y_name]=predict_with_sum(tabel, pre_x_names, pre_y_name)
  tabel=tabel_new.copy()

  # 填充总尿量(缺失特征多的样本)
  pre_x_names=['胸水','腹水','速尿mg','甘露醇ml','碳酸氢钠ml','总入量','出血量']
  pre_y_name='总尿量'
  predict_null=predict_with_labels_dropOtherNA(tabel, pre_x_names, pre_y_name)
  tabel_new.loc[tabel[pre_y_name].isnull(), pre_y_name] = predict_null
  tabel=tabel_new.copy()

  # 速尿mg
  pre_x_names=['胸水','腹水','总尿量','甘露醇ml','碳酸氢钠ml','总入量','出血量']
  pre_y_name='速尿mg'
  predict_null=predict_with_labels_dropOtherNA(tabel, pre_x_names, pre_y_name)
  tabel_new.loc[tabel[pre_y_name].isnull(), pre_y_name] = predict_null
  tabel=tabel_new.copy()

  

  print('tabel_new:\n',tabel_new.info(verbose=True, null_counts=True))
  #  保存填充后的值到总表
  all_data.loc[:,features] = tabel_new
  # 改变列名 术式(经典1背驮2)->术式(经典1背驮0)
  all_data=all_data.rename(columns={'术式(经典1背驮2)':'术式(经典1背驮0)'})
  # 各期尿量(去除)
  all_data.drop( ['I期尿量','II期尿量','III期尿量'],axis = 1,inplace=True)
  # print("all_data columns name:\n",all_data.columns[10:])
  return all_data

def do_data_process(all_data,all_labels):
 train_data = all_data.copy(deep=True)
 tabel1_features=['年龄','身高','体重','BMI','乙肝携带']#序列号和血型对预测基本信息没有帮助
 train_data,all_labels = fill_tabel1(train_data,tabel1_features,all_labels)
#  print("data_index:\n{}\nlabel_index:\n{}".format(train_data.index,all_labels.index))
 tabel2_features=['术式(经典1背驮2)','手术时间min','无肝期时间min','热缺血时间min','冷缺血时间min','红细胞','血浆','自体血','4%白蛋白','2%白蛋白','纯白蛋白g','NS','LR','万汶','佳乐施','总入量','出血量','胸水','腹水','总尿量','I期尿量','II期尿量','III期尿量','速尿mg','甘露醇ml','碳酸氢钠ml','纤维蛋白原g','凝血酶原复合物U','VII因子','氨甲环酸g/h','氨甲环酸入壶g','去甲肾上腺素维持','去甲肾上腺素出室','肾上腺素维持','肾上腺素出室','多巴胺维持mg/h','多巴胺出室','开放时阿托品','开放时最低心率','开放时最低SBP','开放时最低DBP','开放时最低MBP','再灌注后综合征','切脾','肝肾联合移植','特利加压素ml/h']
 train_data = fill_tabel2(train_data,tabel2_features)

#这个暂时不能删, 因为之后PH-150两个都用到了
 all_data = train_data.copy(deep=True)
 medical_data = train_data.copy(deep=True)
 #原代码变量保存end
#  print('medical_data:\n',medical_data.info(verbose=True, null_counts=True))


#  print('-----------------tabel1一般情况------------------')
# #  随机森林辅助中位数填补
#  tabel1=medical_data[['年龄','身高','体重','BMI','乙肝携带']]#序列号和血型对预测基本信息没有帮助
#  pre_y_names=['身高','体重','BMI']
#  tabel1_new=tabel1.copy()
#  for pre_y_name in pre_y_names:
#    predict_null=predict_with_1_label(tabel1, '年龄', pre_y_name)
#    tabel1_new.loc[tabel1[pre_y_name].isnull(), pre_y_name] = predict_null
# #  保存填充后的值
#  tabel1=tabel1_new
#  tabel1_new=tabel1.copy()
# #  乙肝携带所有可能的值填充（Assigning All Possible values of the Attribute）
#  data_isnull = tabel1.loc[tabel1['乙肝携带'].isnull()]
#  data_isnull_idx=tabel1.loc[tabel1['乙肝携带'].isnull()].index
#  data_isnull_filled0=data_isnull.fillna({'乙肝携带':0})
#  data_isnull_filled1=data_isnull.fillna({'乙肝携带':1})
#  tabel1=tabel1.drop(index=data_isnull_idx)
#  tabel1_new = pd.concat([tabel1, data_isnull_filled0, data_isnull_filled1], axis=0, sort=False)
#  print('tabel1:\n',pd.DataFrame(tabel1_new).info())


#  data_notnull = tabel1.loc[tabel1['身高'].notnull()]
#  data_isnull = tabel1.loc[tabel1['身高'].isnull()].drop( ['身高'],axis = 1)
#  data_notnull_filled=data_notnull.fillna(tabel1.median())
# #  data_isnull_filled=data_isnull.fillna(tabel1.median().drop( ['身高']))
#  data_isnull_filled=data_isnull[['年龄']].fillna(tabel1.median()['年龄'])
#  # 随机森林填充
# #  X = data_notnull_filled.drop(['身高'],axis = 1)
#  X = np.array(data_notnull_filled['年龄']).reshape(-1, 1)
#  Y =data_notnull_filled['身高']
#  rfr = RandomForestRegressor(n_estimators=100, n_jobs=-1)
#  rfr.fit(X, Y)
#  predict_null = rfr.predict(np.array(data_isnull_filled['年龄']).reshape(-1, 1))
#  print("data_isnull_filled(X_pre):\n{}\npredict_null(Y_pre):\n{}".format(data_isnull_filled,predict_null))
 
#  tabel1_new.loc[tabel1['身高'].isnull(), '身高'] = predict_null


 print('-----------------随机森林填充缺失率60%-50%------------------')
 print("-----------------------PH-150")
 # print(medical_data.corr())
 # print (train_data.info())
 # print(train_data.head())
 PH150_notnull = all_data.loc[all_data['PH-150'].notnull()]
 #print(PH150_notnull)
 # 使用随机森林时候，不能存在缺失值，否则会出现以下错误：ValueError: Input contains NaN, infinity or a value too large for dtype('float32').
 # 所以在选用PH-150_notnull（）作为随机森林的训练数据时，drop掉出现空的数据，因为PH-150_isnull（可以理解为预测数据）也存在其他属性为空的值，所以采用线性插值来填补数据
 PH150_notnull = PH150_notnull.interpolate()
 PH150_notnull = PH150_notnull.fillna(method='pad')
 PH150_notnull = PH150_notnull.fillna(method='bfill')
 print (PH150_notnull.info())
 PH150_isnull = all_data.loc[all_data['PH-150'].isnull()]
 #print(PH150_isnull.info())
 del PH150_isnull['PH-150']
 PH150_isnull = PH150_isnull.interpolate()
 PH150_isnull = PH150_isnull.fillna(method='pad')
 PH150_isnull = PH150_isnull.fillna(method='bfill')
 PH150_isnull = PH150_isnull.fillna(0.01)

 print(PH150_isnull.info())

 #print(medical_data.head())
 X = PH150_notnull.drop(['PH-150'],axis = 1)
 #print(X.shape)
 Y =PH150_notnull['PH-150']
 #print(len(Y))
 rfr = RandomForestRegressor(n_estimators=600, n_jobs=-1)
 rfr.fit(X, Y)
 predict_PH150 = rfr.predict(PH150_isnull)
#  print("train_data:\n{}\npredict_PH150 len:\n{}".format(train_data.loc[all_data['PH-150'].isnull(), 'PH-150'].shape,predict_PH150.shape))
 train_data.loc[all_data['PH-150'].isnull(), 'PH-150'] = predict_PH150
 print(pd.DataFrame(train_data).info())

 # ------------------------------处理K-150
 print("-----------------------K-150")
 train_data = pd.DataFrame(train_data)
 print(train_data.info())
 data = pd.DataFrame(train_data)
 K150_notnull = data.loc[medical_data['K-150'].notnull()]
 # print(K150_notnull)
 K150_notnull = K150_notnull.interpolate()
 K150_notnull = K150_notnull.fillna(method='pad')
 K150_notnull = K150_notnull.fillna(method='bfill')
 print(K150_notnull.info())
 K150_isnull = data.loc[data['K-150'].isnull()]
 # print(K150_isnull.info())
 del K150_isnull['K-150']
 K150_isnull = K150_isnull.interpolate()
 K150_isnull = K150_isnull.fillna(method='pad')
 K150_isnull = K150_isnull.fillna(method='bfill')
 K150_isnull = K150_isnull.fillna(0.01)

 print(K150_isnull.info())

 # print(medical_data.head())
 X = K150_notnull.drop(['K-150'], axis=1)
 # print(X.shape)
 Y = K150_notnull['K-150']
 # print(len(Y))
 rfr = RandomForestRegressor(n_estimators=600, n_jobs=-1)
 rfr.fit(X, Y)
 predict_K150 = rfr.predict(K150_isnull)
 train_data.loc[data['K-150'].isnull(), 'K-150'] = predict_K150
 print(pd.DataFrame(train_data).info())

 # ------------------------------处理Na-150
 print("-----------------------Na-150")
 train_data = pd.DataFrame(train_data)
 print(train_data.info())
 data = pd.DataFrame(train_data)
 Na150_notnull = data.loc[medical_data['Na-150'].notnull()]
 # print(Na150_notnull)
 Na150_notnull = Na150_notnull.interpolate()
 Na150_notnull = Na150_notnull.fillna(method='pad')
 Na150_notnull = Na150_notnull.fillna(method='bfill')
 print(Na150_notnull.info())
 Na150_isnull = data.loc[data['Na-150'].isnull()]
 # print(Na150_isnull.info())
 del Na150_isnull['Na-150']
 Na150_isnull = Na150_isnull.interpolate()
 Na150_isnull = Na150_isnull.fillna(method='pad')
 Na150_isnull = Na150_isnull.fillna(method='bfill')
 Na150_isnull = Na150_isnull.fillna(0.01)

 print(Na150_isnull.info())

 # print(medical_data.head())
 X = Na150_notnull.drop(['Na-150'], axis=1)
 # print(X.shape)
 Y = Na150_notnull['Na-150']
 # print(len(Y))
 rfr = RandomForestRegressor(n_estimators=600, n_jobs=-1)
 rfr.fit(X, Y)
 predict_Na150 = rfr.predict(Na150_isnull)
 train_data.loc[data['Na-150'].isnull(), 'Na-150'] = predict_Na150
 print(pd.DataFrame(train_data).info())

 # ------------------------------处理Glu-150
 print("-----------------------Glu-150")
 train_data = pd.DataFrame(train_data)
 print(train_data.info())
 data = pd.DataFrame(train_data)
 Glu150_notnull = data.loc[medical_data['Glu-150'].notnull()]
 # print(Glu150_notnull)
 Glu150_notnull = Glu150_notnull.interpolate()
 Glu150_notnull = Glu150_notnull.fillna(method='pad')
 Glu150_notnull = Glu150_notnull.fillna(method='bfill')
 print(Glu150_notnull.info())
 Glu150_isnull = data.loc[data['Glu-150'].isnull()]
 # print(Glu150_isnull.info())
 del Glu150_isnull['Glu-150']
 Glu150_isnull = Glu150_isnull.interpolate()
 Glu150_isnull = Glu150_isnull.fillna(method='pad')
 Glu150_isnull = Glu150_isnull.fillna(method='bfill')
 Glu150_isnull = Glu150_isnull.fillna(0.01)

 print(Glu150_isnull.info())

 # print(medical_data.head())
 X = Glu150_notnull.drop(['Glu-150'], axis=1)
 # print(X.shape)
 Y = Glu150_notnull['Glu-150']
 # print(len(Y))
 rfr = RandomForestRegressor(n_estimators=600, n_jobs=-1)
 rfr.fit(X, Y)
 predict_Glu150 = rfr.predict(Glu150_isnull)
 train_data.loc[data['Glu-150'].isnull(), 'Glu-150'] = predict_Glu150
 print(pd.DataFrame(train_data).info())

 # ------------------------------处理Lac-150
 print("-----------------------Lac-150")
 train_data = pd.DataFrame(train_data)
 print(train_data.info())
 data = pd.DataFrame(train_data)
 Lac150_notnull = data.loc[medical_data['Lac-150'].notnull()]
 # print(Lac150_notnull)
 Lac150_notnull = Lac150_notnull.interpolate()
 Lac150_notnull = Lac150_notnull.fillna(method='pad')
 Lac150_notnull = Lac150_notnull.fillna(method='bfill')
 print(Lac150_notnull.info())
 Lac150_isnull = data.loc[data['Lac-150'].isnull()]
 # print(Lac150_isnull.info())
 del Lac150_isnull['Lac-150']
 Lac150_isnull = Lac150_isnull.interpolate()
 Lac150_isnull = Lac150_isnull.fillna(method='pad')
 Lac150_isnull = Lac150_isnull.fillna(method='bfill')
 Lac150_isnull = Lac150_isnull.fillna(0.01)

 print(Lac150_isnull.info())

 # print(medical_data.head())
 X = Lac150_notnull.drop(['Lac-150'], axis=1)
 # print(X.shape)
 Y = Lac150_notnull['Lac-150']
 # print(len(Y))
 rfr = RandomForestRegressor(n_estimators=600, n_jobs=-1)
 rfr.fit(X, Y)
 predict_Lac150 = rfr.predict(Lac150_isnull)
 train_data.loc[data['Lac-150'].isnull(), 'Lac-150'] = predict_Lac150
 print(pd.DataFrame(train_data).info())

 # ------------------------------处理PO2-150
 print("-----------------------PO2-150")
 train_data = pd.DataFrame(train_data)
 print(train_data.info())
 data = pd.DataFrame(train_data)
 PO2150_notnull = data.loc[medical_data['PO2-150'].notnull()]
 # print(PO2150_notnull)
 PO2150_notnull = PO2150_notnull.interpolate()
 PO2150_notnull = PO2150_notnull.fillna(method='pad')
 PO2150_notnull = PO2150_notnull.fillna(method='bfill')
 print(PO2150_notnull.info())
 PO2150_isnull = data.loc[data['PO2-150'].isnull()]
 # print(PO2150_isnull.info())
 del PO2150_isnull['PO2-150']
 PO2150_isnull = PO2150_isnull.interpolate()
 PO2150_isnull = PO2150_isnull.fillna(method='pad')
 PO2150_isnull = PO2150_isnull.fillna(method='bfill')
 PO2150_isnull = PO2150_isnull.fillna(0.01)

 print(PO2150_isnull.info())

 # print(medical_data.head())
 X = PO2150_notnull.drop(['PO2-150'], axis=1)
 # print(X.shape)
 Y = PO2150_notnull['PO2-150']
 # print(len(Y))
 rfr = RandomForestRegressor(n_estimators=600, n_jobs=-1)
 rfr.fit(X, Y)
 predict_PO2150 = rfr.predict(PO2150_isnull)
 train_data.loc[data['PO2-150'].isnull(), 'PO2-150'] = predict_PO2150
 print(pd.DataFrame(train_data).info())

 # ------------------------------处理Ca-150
 print("-----------------------Ca-150")
 train_data = pd.DataFrame(train_data)
 print(train_data.info())
 data = pd.DataFrame(train_data)
 Ca150_notnull = data.loc[medical_data['Ca-150'].notnull()]
 # print(Ca150_notnull)
 Ca150_notnull = Ca150_notnull.interpolate()
 Ca150_notnull = Ca150_notnull.fillna(method='pad')
 Ca150_notnull = Ca150_notnull.fillna(method='bfill')
 print(Ca150_notnull.info())
 Ca150_isnull = data.loc[data['Ca-150'].isnull()]
 # print(Ca150_isnull.info())
 del Ca150_isnull['Ca-150']
 Ca150_isnull = Ca150_isnull.interpolate()
 Ca150_isnull = Ca150_isnull.fillna(method='pad')
 Ca150_isnull = Ca150_isnull.fillna(method='bfill')
 Ca150_isnull = Ca150_isnull.fillna(0.01)

 print(Ca150_isnull.info())

 # print(medical_data.head())
 X = Ca150_notnull.drop(['Ca-150'], axis=1)
 # print(X.shape)
 Y = Ca150_notnull['Ca-150']
 # print(len(Y))
 rfr = RandomForestRegressor(n_estimators=600, n_jobs=-1)
 rfr.fit(X, Y)
 predict_Ca150 = rfr.predict(Ca150_isnull)
 train_data.loc[data['Ca-150'].isnull(), 'Ca-150'] = predict_Ca150
 print(pd.DataFrame(train_data).info())

 # ------------------------------处理PCO2-150
 print("-----------------------PCO2-150")
 train_data = pd.DataFrame(train_data)
 print(train_data.info())
 data = pd.DataFrame(train_data)
 PCO2150_notnull = data.loc[medical_data['PCO2-150'].notnull()]
 # print(PCO2150_notnull)
 PCO2150_notnull = PCO2150_notnull.interpolate()
 PCO2150_notnull = PCO2150_notnull.fillna(method='pad')
 PCO2150_notnull = PCO2150_notnull.fillna(method='bfill')
 print(PCO2150_notnull.info())
 PCO2150_isnull = data.loc[data['PCO2-150'].isnull()]
 # print(PCO2150_isnull.info())
 del PCO2150_isnull['PCO2-150']
 PCO2150_isnull = PCO2150_isnull.interpolate()
 PCO2150_isnull = PCO2150_isnull.fillna(method='pad')
 PCO2150_isnull = PCO2150_isnull.fillna(method='bfill')
 PCO2150_isnull = PCO2150_isnull.fillna(0.01)

 print(PCO2150_isnull.info())

 # print(medical_data.head())
 X = PCO2150_notnull.drop(['PCO2-150'], axis=1)
 # print(X.shape)
 Y = PCO2150_notnull['PCO2-150']
 # print(len(Y))
 rfr = RandomForestRegressor(n_estimators=600, n_jobs=-1)
 rfr.fit(X, Y)
 predict_PCO2150 = rfr.predict(PCO2150_isnull)
 train_data.loc[data['PCO2-150'].isnull(), 'PCO2-150'] = predict_PCO2150
 print(pd.DataFrame(train_data).info())

 # ------------------------------处理PR-7
 print("-----------------------PR-7")
 train_data = pd.DataFrame(train_data)
 print(train_data.info())
 data = pd.DataFrame(train_data)
 PR7_notnull = data.loc[medical_data['PR-7'].notnull()]
 # print(PR7_notnull)
 PR7_notnull = PR7_notnull.interpolate()
 PR7_notnull = PR7_notnull.fillna(method='pad')
 PR7_notnull = PR7_notnull.fillna(method='bfill')
 print(PR7_notnull.info())
 PR7_isnull = data.loc[data['PR-7'].isnull()]
 # print(PR7_isnull.info())
 del PR7_isnull['PR-7']
 PR7_isnull = PR7_isnull.interpolate()
 PR7_isnull = PR7_isnull.fillna(method='pad')
 PR7_isnull = PR7_isnull.fillna(method='bfill')
 PR7_isnull = PR7_isnull.fillna(0.01)

 print(PR7_isnull.info())

 # print(medical_data.head())
 X = PR7_notnull.drop(['PR-7'], axis=1)
 # print(X.shape)
 Y = PR7_notnull['PR-7']
 # print(len(Y))
 rfr = RandomForestRegressor(n_estimators=600, n_jobs=-1)
 rfr.fit(X, Y)
 predict_PR7 = rfr.predict(PR7_isnull)
 train_data.loc[data['PR-7'].isnull(), 'PR-7'] = predict_PR7
 print(pd.DataFrame(train_data).info())

 # ------------------------------处理INR-7
 print("-----------------------INR-7")
 train_data = pd.DataFrame(train_data)
 print(train_data.info())
 data = pd.DataFrame(train_data)
 INR7_notnull = data.loc[medical_data['INR-7'].notnull()]
 # print(INR7_notnull)
 INR7_notnull = INR7_notnull.interpolate()
 INR7_notnull = INR7_notnull.fillna(method='pad')
 INR7_notnull = INR7_notnull.fillna(method='bfill')
 print(INR7_notnull.info())
 INR7_isnull = data.loc[data['INR-7'].isnull()]
 # print(INR7_isnull.info())
 del INR7_isnull['INR-7']
 INR7_isnull = INR7_isnull.interpolate()
 INR7_isnull = INR7_isnull.fillna(method='pad')
 INR7_isnull = INR7_isnull.fillna(method='bfill')
 INR7_isnull = INR7_isnull.fillna(0.01)

 print(INR7_isnull.info())

 # print(medical_data.head())
 X = INR7_notnull.drop(['INR-7'], axis=1)
 # print(X.shape)
 Y = INR7_notnull['INR-7']
 # print(len(Y))
 rfr = RandomForestRegressor(n_estimators=600, n_jobs=-1)
 rfr.fit(X, Y)
 predict_INR7 = rfr.predict(INR7_isnull)
 train_data.loc[data['INR-7'].isnull(), 'INR-7'] = predict_INR7
 print(pd.DataFrame(train_data).info())

 # ------------------------------处理FBG-7
 print("-----------------------FBG-7")
 train_data = pd.DataFrame(train_data)
 print(train_data.info())
 data = pd.DataFrame(train_data)
 FBG7_notnull = data.loc[medical_data['FBG-7'].notnull()]
 # print(FBG7_notnull)
 FBG7_notnull = FBG7_notnull.interpolate()
 FBG7_notnull = FBG7_notnull.fillna(method='pad')
 FBG7_notnull = FBG7_notnull.fillna(method='bfill')
 print(FBG7_notnull.info())
 FBG7_isnull = data.loc[data['FBG-7'].isnull()]
 # print(FBG7_isnull.info())
 del FBG7_isnull['FBG-7']
 FBG7_isnull = FBG7_isnull.interpolate()
 FBG7_isnull = FBG7_isnull.fillna(method='pad')
 FBG7_isnull = FBG7_isnull.fillna(method='bfill')
 FBG7_isnull = FBG7_isnull.fillna(0.01)

 print(FBG7_isnull.info())

 # print(medical_data.head())
 X = FBG7_notnull.drop(['FBG-7'], axis=1)
 # print(X.shape)
 Y = FBG7_notnull['FBG-7']
 # print(len(Y))
 rfr = RandomForestRegressor(n_estimators=600, n_jobs=-1)
 rfr.fit(X, Y)
 predict_FBG7 = rfr.predict(FBG7_isnull)
 train_data.loc[data['FBG-7'].isnull(), 'FBG-7'] = predict_FBG7
 print(pd.DataFrame(train_data).info())

 # ------------------------------处理APTT-7
 print("-----------------------APTT-7")
 train_data = pd.DataFrame(train_data)
 print(train_data.info())
 data = pd.DataFrame(train_data)
 APTT7_notnull = data.loc[medical_data['APTT-7'].notnull()]
 # print(APTT7_notnull)
 APTT7_notnull = APTT7_notnull.interpolate()
 APTT7_notnull = APTT7_notnull.fillna(method='pad')
 APTT7_notnull = APTT7_notnull.fillna(method='bfill')
 print(APTT7_notnull.info())
 APTT7_isnull = data.loc[data['APTT-7'].isnull()]
 # print(APTT7_isnull.info())
 del APTT7_isnull['APTT-7']
 APTT7_isnull = APTT7_isnull.interpolate()
 APTT7_isnull = APTT7_isnull.fillna(method='pad')
 APTT7_isnull = APTT7_isnull.fillna(method='bfill')
 APTT7_isnull = APTT7_isnull.fillna(0.01)

 print(APTT7_isnull.info())

 # print(medical_data.head())
 X = APTT7_notnull.drop(['APTT-7'], axis=1)
 # print(X.shape)
 Y = APTT7_notnull['APTT-7']
 # print(len(Y))
 rfr = RandomForestRegressor(n_estimators=600, n_jobs=-1)
 rfr.fit(X, Y)
 predict_APTT7 = rfr.predict(APTT7_isnull)
 train_data.loc[data['APTT-7'].isnull(), 'APTT-7'] = predict_APTT7
 print(pd.DataFrame(train_data).info())

 # ------------------------------处理PT-7
 print("-----------------------PT-7")
 train_data = pd.DataFrame(train_data)
 print(train_data.info())
 data = pd.DataFrame(train_data)
 PT7_notnull = data.loc[medical_data['PT-7'].notnull()]
 # print(PT7_notnull)
 PT7_notnull = PT7_notnull.interpolate()
 PT7_notnull = PT7_notnull.fillna(method='pad')
 PT7_notnull = PT7_notnull.fillna(method='bfill')
 print(PT7_notnull.info())
 PT7_isnull = data.loc[data['PT-7'].isnull()]
 # print(PT7_isnull.info())
 del PT7_isnull['PT-7']
 PT7_isnull = PT7_isnull.interpolate()
 PT7_isnull = PT7_isnull.fillna(method='pad')
 PT7_isnull = PT7_isnull.fillna(method='bfill')
 PT7_isnull = PT7_isnull.fillna(0.01)

 print(PT7_isnull.info())

 # print(medical_data.head())
 X = PT7_notnull.drop(['PT-7'], axis=1)
 # print(X.shape)
 Y = PT7_notnull['PT-7']
 # print(len(Y))
 rfr = RandomForestRegressor(n_estimators=600, n_jobs=-1)
 rfr.fit(X, Y)
 predict_PT7 = rfr.predict(PT7_isnull)
 train_data.loc[data['PT-7'].isnull(), 'PT-7'] = predict_PT7
 print(pd.DataFrame(train_data).info())

 # ------------------------------处理PA-7
 print("-----------------------PA-7")
 train_data = pd.DataFrame(train_data)
 print(train_data.info())
 data = pd.DataFrame(train_data)
 PA7_notnull = data.loc[medical_data['PA-7'].notnull()]
 # print(PA7_notnull)
 PA7_notnull = PA7_notnull.interpolate()
 PA7_notnull = PA7_notnull.fillna(method='pad')
 PA7_notnull = PA7_notnull.fillna(method='bfill')
 print(PA7_notnull.info())
 PA7_isnull = data.loc[data['PA-7'].isnull()]
 # print(PA7_isnull.info())
 del PA7_isnull['PA-7']
 PA7_isnull = PA7_isnull.interpolate()
 PA7_isnull = PA7_isnull.fillna(method='pad')
 PA7_isnull = PA7_isnull.fillna(method='bfill')
 PA7_isnull = PA7_isnull.fillna(0.01)

 print(PA7_isnull.info())

 # print(medical_data.head())
 X = PA7_notnull.drop(['PA-7'], axis=1)
 # print(X.shape)
 Y = PA7_notnull['PA-7']
 # print(len(Y))
 rfr = RandomForestRegressor(n_estimators=600, n_jobs=-1)
 rfr.fit(X, Y)
 predict_PA7 = rfr.predict(PA7_isnull)
 train_data.loc[data['PA-7'].isnull(), 'PA-7'] = predict_PA7
 print(pd.DataFrame(train_data).info())



 print('____________使用均值填充或者插值法补全缺失率低的缺失值___________________')
 print('———————————进行线性插值————————————')
 train_data = pd.DataFrame(train_data)
 medical_data_fill_na = train_data.interpolate()  # 线性插值
 medical_data_fill_na = medical_data_fill_na.fillna(method='pad')
 medical_data_fill_na = medical_data_fill_na.fillna(method='bfill')

 print('———————————————————将预处理的结果进行保存————————————————————————')
 medical_data_fill_na.to_csv('./data/fill_NA_features.csv', index=False)
 all_labels.to_csv('./data/fill_labels.csv', index=False)


if __name__=='__main__':
  # missing_df = missing_values_table(medical_data)
  missing_values_table(medical_data)
  remaining_data=get_remaining_features(medical_data)
  need_dealing_features=get_need_dealing_features(remaining_data)
  # show_missing(need_dealing_features)
  do_data_process(remaining_data,medical_labels)


