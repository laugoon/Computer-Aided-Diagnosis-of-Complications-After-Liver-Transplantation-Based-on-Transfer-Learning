# coding: utf-8

import pandas as pd
import numpy as np
import scipy.linalg
# import sklearn.metrics
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

from sklearn import metrics
# import  pickle
from sklearn.metrics import f1_score
# from collections import Counter
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

# def knn_fit_predict( X_train, Y_train, X_test, Y_test):
#     print('-------##K近邻算法-----------')
#     clf = KNeighborsClassifier(n_neighbors=15)
#     clf.fit(X_train, Y_train)
#     y_pred = clf.predict(X_test)
#     f1 = sklearn.metrics.f1_score(Y_test, y_pred)
#     return y_pred, f1

# 对同一个训练集的单个label训练+交叉验证
def train_knn_1label(all_train_lable,all_test_lable,train_data,test_data,label_idx):

    print('------------{}--------------'.format(y_columns[label_idx]))
    # train_data_lable1 = train_data_lable[y_columns[label_idx]]
    # test_data_lable1 = test_data_lable[y_columns[label_idx]]
    # ypre, f1 = knn_fit_predict(train_data, train_data_lable1, test_data, test_data_lable1)

    # 拼接源域目标域, 并取对应任务目标的一列y值
    X = np.vstack((train_data, test_data))
    Y = np.hstack((all_train_lable[y_columns[label_idx]], all_test_lable[y_columns[label_idx]]))
    # # 查看形状
    # print("shape of X:{}\nshape of Y:{}".format(X.shape,Y.shape))
    # print("X:{}\nY:{}".format(X,Y))
    #训练模型
    # svm = SVC(kernel='rbf', C=100.0,gamma=100.0, random_state=0)
    clf = KNeighborsClassifier(n_neighbors=15)
    
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
        clone_clf.fit(X_train_folds, y_train_folds)
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

# 对同一个训练集的单个label训练+交叉验证+上采样
def train_1label_TCA_KNN_ROS(Xs,all_ys,Xt,all_yt,label_idx):

    print('------------{}--------------'.format(y_columns[label_idx]))
    
    # 取对应任务目标的一列y值
    Ys = all_ys[:,label_idx]
    Yt = all_yt[:,label_idx]

    
    #训练模型
    # svm = SVC(kernel='rbf', C=100.0,gamma=100.0, random_state=0)
    clf = KNeighborsClassifier(n_neighbors=15)
    
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
        clone_clf.fit(X_res, y_res)
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


def TCA_KNN_all_label(label_num=9):
#     train_data = pd.read_csv('./data/TCA_source_data.csv')
#     test_data = pd.read_csv('./data/TCA_target_data.csv')
    # recall = np.zeros(label_num) 
    f1 = np.zeros(label_num) 
    for i in range(label_num):
        # recall[i], f1[i]=train_SVM_cross_val(train_data_lable,test_data_lable,train_data,test_data,i)
        f1[i]=train_knn_1label(train_data_lable,test_data_lable,train_data,test_data,i)
    # 保存预测后结果
    # recall = pd.DataFrame(recall)
    f1 = pd.DataFrame(f1)
    # recall.to_csv('./data/12.16features_TCA+SVM_cross_val_recall.csv', index=False)
    f1.to_csv('./data/{}features_TCA+KNN_cross_val_f1.csv'.format(curtime), index=False)
    print('Performance saved')

def TCA_KNN_ROS_train(label_num=9):
#     train_data = pd.read_csv('./data/TCA_source_data.csv')
#     test_data = pd.read_csv('./data/TCA_target_data.csv')
    # recall = np.zeros(label_num) 
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
        recall[i], f1[i], prc[i]=train_1label_TCA_KNN_ROS(xs,ys, xt,yt,i)

    '''标准数据集划分end'''	
    # 保存预测后结果
    recall = pd.DataFrame(recall)
    f1 = pd.DataFrame(f1)
    prc = pd.DataFrame(prc)
    # global_shap = pd.DataFrame(global_shap)
    performance = pd.concat([f1, recall, prc], axis=1, sort=False)
    # 设置表头
    performance.columns = ['F1','Recall','Precision']
    performance.to_csv('./data/imbalance_experiment/{}TCA+KNN_ROS_performance.csv'.format(curtime), index=False)

    print('KNN Performance saved')



    # f1 = np.zeros(label_num) 
    # for i in range(label_num):
    #     # recall[i], f1[i]=train_SVM_cross_val(train_data_lable,test_data_lable,train_data,test_data,i)
    #     f1[i]=train_knn_1label(train_data_lable,test_data_lable,train_data,test_data,i)
    # # 保存预测后结果
    # # recall = pd.DataFrame(recall)
    # f1 = pd.DataFrame(f1)
    # # recall.to_csv('./data/12.16features_TCA+SVM_cross_val_recall.csv', index=False)
    # f1.to_csv('./data/{}features_TCA+KNN_cross_val_f1.csv'.format(curtime), index=False)
    # print('Performance saved')

if __name__ == '__main__':
    TCA_KNN_all_label(9)
    # TCA_KNN_ROS_train(9)#训练集上采样
    

    


