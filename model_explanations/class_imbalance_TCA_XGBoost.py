import pandas as pd
import numpy as np
import scipy.linalg
import sklearn.metrics
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import pandas as pd
from xgboost.sklearn import XGBClassifier
# import xgboost as xgb
from xgboost import plot_importance
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn import  metrics

from sklearn.metrics import f1_score
#added
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
import time
import sys
sys.path.append('./')
from data_process_shap.TCA_shap import TCA,TCA_data_read
from collections import Counter
from imblearn.over_sampling import RandomOverSampler

curtime = time.strftime('%Y-%m-%d',time.localtime(time.time()))

train_data = pd.read_csv('data\TCA_source_data.csv')
test_data = pd.read_csv('data\TCA_target_data.csv')
# train_data_lable = pd.read_csv('data\source_data_lable.csv')
# test_data_lable = pd.read_csv('data\\target_data_lable.csv')
#预测标签
# train_data_lable = pd.read_csv('./data/labels.csv')
train_data_lable = pd.read_csv('./data/fill_labels.csv')
test_data_lable = train_data_lable.copy(deep=True)
y_columns=['胸腔积液','凝血功能指标','转氨酶指标','胆红素指标','急性肺损伤','术后出血','术后感染','胆道并发症','原发性移植肝无功能']

# 对同一个训练集的单个label训练+交叉验证
def train_xgb_1label(train_lable,test_lable,train_data,test_data,label_name):

    print('------------{}--------------'.format(label_name))
    # 拼接源域目标域
    X = np.vstack((train_data, test_data))
    Y = np.hstack((train_lable, test_lable))
    # # 查看形状
    # print("shape of X:{}\nshape of Y:{}".format(X.shape,Y.shape))
    # print("X:{}\nY:{}".format(X,Y))
    # clf = KNeighborsClassifier(n_neighbors=15)
    # 模型参数
    clf = XGBClassifier(
        # silent=1,  # 设置成1则没有运行信息输出，最好是设置为0.是否在运行升级时打印消息。
        # nthread=4,# cpu 线程数 默认最大
        learning_rate=0.1,  # 如同学习率
        min_child_weight=1,
        # 这个参数默认是 1，是每个叶子里面 h 的和至少是多少，对正负样本不均衡时的 0-1 分类而言
        # ，假设 h 在 0.01 附近，min_child_weight 为 1 意味着叶子节点中最少需要包含 100 个样本。
        # 这个参数非常影响结果，控制叶子节点中二阶导的和的最小值，该参数值越小，越容易 overfitting。
        max_depth=6,  # 构建树的深度，越大越容易过拟合
        gamma=0,  # 树的叶子节点上作进一步分区所需的最小损失减少,越大越保守，一般0.1、0.2这样子。
        subsample=1,  # 随机采样训练样本 训练实例的子采样比
        max_delta_step=0,  # 最大增量步长，我们允许每个树的权重估计。
        colsample_bytree=1,  # 生成树时进行的列采样
        reg_lambda=1,  # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。

        scale_pos_weight=1,  # 如果取值大于0的话，在类别样本不平衡的情况下有助于快速收敛。平衡正负权重

        n_estimators=60,  # 树的个数
        seed=1000  # 随机种子

    )
    


    # 分层采样+K折交叉
    n_splits=3
    skfolds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # recall=0
    f1=0

    for train_index, test_index in skfolds.split(X, Y):
        clone_clf = clone(clf)
        X_train_folds = X[train_index]
        y_train_folds = (Y[train_index])
        X_test_fold = X[test_index]
        y_test_fold = (Y[test_index])
        #训练, 预测
        clone_clf.fit(X_train_folds, y_train_folds, eval_metric='auc')#在fit时的评判指标(auc)代表的是ROC曲线的面积
        y_pred = clone_clf.predict(X_test_fold)
        # recall+=metrics.recall_score(y_test_fold, y_pred)
        f1 += f1_score(y_test_fold, y_pred)#循环中加, 循环结束后再算平均
    # recall /= n_splits
    f1 /= n_splits
    # print('召回率: %.2f' % recall)
    # return recall, f1
    print('F1值:%.2f' % f1)
    # print(ypre)
    return f1

# XGBoost上采样
def train_1label_TCA_XGBoost_ROS(Xs,all_ys,Xt,all_yt,label_idx):
    print('------------{}--------------'.format(y_columns[label_idx]))
    # 取对应任务目标的一列y值
    Ys = all_ys[:,label_idx]
    Yt = all_yt[:,label_idx]

#    # print('------------{}--------------'.format(label_name))
    # # 拼接源域目标域
    # X = np.vstack((train_data, test_data))
    # Y = np.hstack((train_lable, test_lable))
    # # 查看形状
    # print("shape of X:{}\nshape of Y:{}".format(X.shape,Y.shape))
    # print("X:{}\nY:{}".format(X,Y))
    # clf = KNeighborsClassifier(n_neighbors=15)
    # 模型参数
    clf = XGBClassifier(
        # silent=1,  # 设置成1则没有运行信息输出，最好是设置为0.是否在运行升级时打印消息。
        # nthread=4,# cpu 线程数 默认最大
        learning_rate=0.1,  # 如同学习率
        min_child_weight=1,
        # 这个参数默认是 1，是每个叶子里面 h 的和至少是多少，对正负样本不均衡时的 0-1 分类而言
        # ，假设 h 在 0.01 附近，min_child_weight 为 1 意味着叶子节点中最少需要包含 100 个样本。
        # 这个参数非常影响结果，控制叶子节点中二阶导的和的最小值，该参数值越小，越容易 overfitting。
        max_depth=6,  # 构建树的深度，越大越容易过拟合
        gamma=0,  # 树的叶子节点上作进一步分区所需的最小损失减少,越大越保守，一般0.1、0.2这样子。
        subsample=1,  # 随机采样训练样本 训练实例的子采样比
        max_delta_step=0,  # 最大增量步长，我们允许每个树的权重估计。
        colsample_bytree=1,  # 生成树时进行的列采样
        reg_lambda=1,  # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。

        scale_pos_weight=1,  # 如果取值大于0的话，在类别样本不平衡的情况下有助于快速收敛。平衡正负权重

        n_estimators=60,  # 树的个数
        seed=1000  # 随机种子

    )
    


    # 分层采样+K折交叉
    n_splits=3
    skfolds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    recall=0
    f1=0
    prc=0

    for train_index, test_index in skfolds.split(Xt, Yt):
    # for train_index, test_index in skfolds.split(X, Y):
        n_components=50
        
        clone_clf = clone(clf)
        Xt_train_folds = Xt[train_index]
        yt_train_folds = Yt[train_index]
        Xt_test_fold = Xt[test_index]
        yt_test_fold = Yt[test_index]
        # X_train_folds = X[train_index]
        # y_train_folds = (Y[train_index])
        # X_test_fold = X[test_index]
        # y_test_fold = (Y[test_index])
        source_data=Xs
        target_data=Xt_train_folds
        # 迁移
        # TCA降维
        tca = TCA(kernel_type='rbf', dim=n_components, lamb=0.9, gamma=0.5)
        # TCA训练集
        # fit训练集
        Xs_new, Xt_train_new = tca.fit(source_data, target_data)
        # # 训练集与测试集统一用fit_new
        # tca.fit(source_data, target_data)
        # Xs_new= tca.fit_new(source_data, target_data, source_data)
        # Xt_train_new = tca.fit_new(source_data, target_data, target_data)

        #TCA测试集
        Xt_test_new = tca.fit_new(source_data, target_data, Xt_test_fold)

        # 拼接源域目标域的训练集
        x_train_r = np.vstack((Xs_new, Xt_train_new))
        y_train = np.hstack((Ys, yt_train_folds))
        # 背景数据集是迁移之前的训练集(源域+目标域的训练部分)组合
        x_train = np.vstack((source_data, target_data))
        
        counter=Counter(y_train)
        minor_labels_ratio=counter[1]/(counter[0]+counter[1])
        print("labels_ratio:",minor_labels_ratio)
        if(minor_labels_ratio<0.22):
            # 上采样
            sampler = RandomOverSampler(random_state=0)
            X_res, y_res = sampler.fit_resample(x_train_r, y_train)
            print('Original dataset shape %s' % Counter(y_train))
            print('Resampled dataset shape %s' % Counter(y_res))
        else:#没达到阈值则不进行操作
            X_res=x_train_r
            y_res=y_train
        #训练, 预测
        clone_clf.fit(X_res, y_res, eval_metric='auc')#在fit时的评判指标(auc)代表的是ROC曲线的面积
        y_pred = clone_clf.predict(Xt_test_new)
        
        recall+=metrics.recall_score(yt_test_fold, y_pred)
        f1 += f1_score(yt_test_fold, y_pred)
        prc+=metrics.precision_score(yt_test_fold, y_pred)

    recall /= n_splits
    f1 /= n_splits
    prc /= n_splits
    # print('召回率: %.2f' % recall)
    # return recall, f1
    print('F1值:%.2f' % f1)
    # print(ypre)
    # return f1
    return recall, f1, prc
    # return f1

def TCA_XGBoost_all_label(label_num=9):
    f1 = np.zeros(label_num) 
    for i in range(label_num):
        # 将对应目标的一列标签取出
        label_name=y_columns[i]
        train_lable_i = train_data_lable[label_name]
        test_lable_i = test_data_lable[label_name]
        # recall[i], f1[i]=train_SVM_cross_val(train_data_lable,test_data_lable,train_data,test_data,i)
        f1[i]=train_xgb_1label(train_lable_i,test_lable_i,train_data,test_data,label_name)
    # 保存预测后结果
    # recall = pd.DataFrame(recall)
    f1 = pd.DataFrame(f1)
    # f1.to_csv('./data/1.7features_TCA+XGBoost_cross_val_f1.csv', index=False)
    f1.to_csv('./data/{}features_TCA+XGBoost_cross_val_f1.csv'.format(curtime), index=False)
    print('Performance saved')
# 上采样
def TCA_XGBoost_ROS_all_label(label_num=9):
    xs,ys, xt,yt=TCA_data_read()
    # 查看形状
    print("shape of X:{}\nshape of Y:{}".format(xs.shape,ys.shape))
    

    '''标准数据集划分'''
    feature_num=xs.shape[1]
    recall = np.zeros(label_num) 
    f1 = np.zeros(label_num) 
    prc= np.zeros(label_num)
    # global_shap=np.empty(shape=(label_num,feature_num))
    for i in range(label_num):
        # recall[i], f1[i], global_shap[i]=train_1label_TCA_SVM_wrapper_shap(xs,ys, xt,yt,i)

        # 无shap+有precision
        # recall[i], f1[i], prc[i]=train_knn_1label(xs,ys, xt,yt,i)
        recall[i], f1[i], prc[i]=train_1label_TCA_XGBoost_ROS(xs,ys, xt,yt,i)
        # f1[i]=train_xgb_1label(train_lable_i,test_lable_i,train_data,test_data,label_name)


    '''标准数据集划分end'''	
    # 保存预测后结果
    recall = pd.DataFrame(recall)
    f1 = pd.DataFrame(f1)
    prc = pd.DataFrame(prc)
    # global_shap = pd.DataFrame(global_shap)
    performance = pd.concat([f1, recall, prc], axis=1, sort=False)
    # 设置表头
    performance.columns = ['F1','Recall','Precision']
    performance.to_csv('./data/imbalance_experiment/{}TCA+XGBoost_ROS_performance.csv'.format(curtime), index=False)


    # f1 = np.zeros(label_num) 
    # for i in range(label_num):
    #     # 将对应目标的一列标签取出
    #     label_name=y_columns[i]
    #     train_lable_i = train_data_lable[label_name]
    #     test_lable_i = test_data_lable[label_name]
    #     # recall[i], f1[i]=train_SVM_cross_val(train_data_lable,test_data_lable,train_data,test_data,i)
    #     f1[i]=train_xgb_1label(train_lable_i,test_lable_i,train_data,test_data,label_name)
    # # 保存预测后结果
    # # recall = pd.DataFrame(recall)
    # f1 = pd.DataFrame(f1)
    # # f1.to_csv('./data/1.7features_TCA+XGBoost_cross_val_f1.csv', index=False)
    # f1.to_csv('./data/{}features_TCA+XGBoost_cross_val_f1.csv'.format(curtime), index=False)
    print('XGBoost saved')

if __name__ == '__main__':
    # TCA_XGBoost_all_label(9)
    TCA_XGBoost_ROS_all_label(9)



    # # 模型参数
    # clf = XGBClassifier(
    #     # silent=1,  # 设置成1则没有运行信息输出，最好是设置为0.是否在运行升级时打印消息。
    #     # nthread=4,# cpu 线程数 默认最大
    #     learning_rate=0.1,  # 如同学习率
    #     min_child_weight=1,
    #     # 这个参数默认是 1，是每个叶子里面 h 的和至少是多少，对正负样本不均衡时的 0-1 分类而言
    #     # ，假设 h 在 0.01 附近，min_child_weight 为 1 意味着叶子节点中最少需要包含 100 个样本。
    #     # 这个参数非常影响结果，控制叶子节点中二阶导的和的最小值，该参数值越小，越容易 overfitting。
    #     max_depth=6,  # 构建树的深度，越大越容易过拟合
    #     gamma=0,  # 树的叶子节点上作进一步分区所需的最小损失减少,越大越保守，一般0.1、0.2这样子。
    #     subsample=1,  # 随机采样训练样本 训练实例的子采样比
    #     max_delta_step=0,  # 最大增量步长，我们允许每个树的权重估计。
    #     colsample_bytree=1,  # 生成树时进行的列采样
    #     reg_lambda=1,  # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。

    #     scale_pos_weight=1,  # 如果取值大于0的话，在类别样本不平衡的情况下有助于快速收敛。平衡正负权重

    #     n_estimators=60,  # 树的个数
    #     seed=1000  # 随机种子

    # )

