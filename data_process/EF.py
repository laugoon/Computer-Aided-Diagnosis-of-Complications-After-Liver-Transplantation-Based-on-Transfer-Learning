import random

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from lightgbm import LGBMClassifier
from pandas.core.frame import DataFrame
from sklearn.datasets import load_diabetes
from sklearn.ensemble import ExtraTreesRegressor, AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.ensemble import ExtraTreesClassifier, AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from xgboost import XGBClassifier
from catboost import CatBoostRegressor
from catboost import CatBoostClassifier
from evolutionary_forest.utils import get_feature_importance, plot_feature_importance, feature_append
from evolutionary_forest.forest import cross_val_score, EvolutionaryForestRegressor, EvolutionaryForestClassifier

import matplotlib.pyplot as plt
import seaborn as sns






# source_data = pd.read_csv('./data/source_data.csv')
# target_data = pd.read_csv('./data/target_data.csv')
# source_data_lable = pd.read_csv('./data/source_data_lable.csv')
# target_data_lable = pd.read_csv('./data/target_data_lable.csv')

source_data = pd.read_csv('./data/source_data.csv')
target_data = pd.read_csv('./data/target_data.csv')
# source_data_lable = pd.read_csv('./data/labels.csv')
source_data_lable = pd.read_csv('./data/fill_labels.csv')
# 防止浅拷贝问题
target_data_lable = source_data_lable.copy(deep=True)


# #看数据集形状
# print('X:',source_data.shape,target_data.shape,X.shape )


random.seed(0)
np.random.seed(0)

# X, y = load_diabetes(return_X_y=True)
# data = pd.read_csv('./data/do_fill_na.csv')


# data.drop(['序列号','身高', 'Unnamed: 0.1', 'Unnamed: 0'], axis=1, inplace=True)
# # print(data.columns.values)
# #使用train_test_split函数划分数据集(训练集占80%，验证集占20%)

# X = data.iloc[:,0:456]
# Y = data.iloc[:, 456:463]


# print('X:',X)




# 模型训练
# def train_EF(x_train,y_train,x_test,y_test,label_idx):
def train_EF(X,Y,label_idx,dim=50):
    random.seed(0)
    np.random.seed(0)
    # x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
    r = EvolutionaryForestClassifier(max_height=8, normalize=True, select='AutomaticLexicase',
                                    mutation_scheme='weight-plus-cross-global',
                                    gene_num=10, boost_size=100, n_gen=100, base_learner='DT',
                                    verbose=True)
    # r.fit(x_train, y_train[:,0])
    # print(r2_score(y_test[:,0], r.predict(x_test)))
    # r.fit(x_train, y_train[:,label_idx])
    r.fit(X, Y[:,label_idx])
    # # 进行排序(显示前15个)
    # feature_importance_dict = get_feature_importance(r)
    # plot_feature_importance(feature_importance_dict)

    # 利用构造好的新特征(最重要dim个)，改进现有模型的性能
    code_importance_dict = get_feature_importance(r, simple_version=False)
    new_X = feature_append(r, X, list(code_importance_dict.keys())[:dim], only_new_features=True)
    # new_train = feature_append(r, x_train, list(code_importance_dict.keys())[:20], only_new_features=True)
    # new_test = feature_append(r, x_test, list(code_importance_dict.keys())[:20], only_new_features=True)

    # 返回所有构建的新特征
    return new_X
    

#拟合9个并发症目标
def train_all_label_EF(X,Y,data_name):
    # x_train=np.array(x_train)
    # x_test=np.array(x_test)
    # y_train=np.array(y_train)
    # y_test = np.array(y_test)
    X=np.array(X)
    Y=np.array(Y)
    for i in range(9):
        # train_EF(x_train,y_train,x_test,y_test,i)
        new_data=train_EF(X,Y,i,dim=100)
        #保存新特征的数据
        new_data = pd.DataFrame(new_data)
        new_data.to_csv('./data/{}_fity{}.csv'.format(data_name,i), index=False)
        
        print('{} new features saved'.format(i))

if __name__ == '__main__':
    #源域特征重构
    train_all_label_EF(source_data,source_data_lable,'EF_source_data')
    # 目标域特征重构
    train_all_label_EF(target_data,target_data_lable,'EF_target_data')
    


    # # 着重看index=3(术后并发症IIIb)
    # random.seed(0)
    # np.random.seed(0)
    # r = EvolutionaryForestClassifier(max_height=8, normalize=True, select='AutomaticLexicase',
    #                                 mutation_scheme='weight-plus-cross-global',
    #                                 gene_num=10, boost_size=100, n_gen=100, base_learner='DT',
    #                                 verbose=True)
    # r.fit(x_train, y_train[:,3])
    # # 进行排序(显示前15个)
    # feature_importance_dict = get_feature_importance(r)
    # plot_feature_importance(feature_importance_dict)
