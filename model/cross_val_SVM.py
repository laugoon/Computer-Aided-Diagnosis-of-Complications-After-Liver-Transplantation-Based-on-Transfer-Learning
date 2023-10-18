import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn import metrics
import  pickle
from sklearn.metrics import f1_score
# from imblearn.pipeline import make_pipeline
from imblearn.over_sampling import RandomOverSampler,SMOTE,ADASYN
from collections import Counter

#add
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
import time
#add



#预测标签
# train_data_lable = pd.read_csv('./data/labels.csv')
train_data_lable = pd.read_csv('./data/fill_labels.csv')
test_data_lable = train_data_lable.copy(deep=True)
# y_columns = ['所有胸腔积液情况']
y_columns=['胸腔积液','凝血功能指标','转氨酶指标','胆红素指标','急性肺损伤','术后出血','术后感染','胆道并发症','原发性移植肝无功能']
curtime = time.strftime('%Y-%m-%d',time.localtime(time.time()))

# print("train X shape:",train_data.shape)
# print("test X shape:",test_data.shape)
# print("train Y shape:",train_data_lable.shape)
# print("test Y shape:",test_data_lable.shape)

# 对同一个训练集的单个label训练
def train_SVM(all_train_lable,all_test_lable,train_data,test_data,label_idx):
    print('------------{}--------------'.format(y_columns[label_idx]))
    train_data_lable = all_train_lable[y_columns[label_idx]]
    test_data_lable = all_test_lable[y_columns[label_idx]]
    #训练模型
    svm1 = SVC(kernel='rbf', C=100.0,gamma=100.0, random_state=0)
    svm1.fit(train_data, train_data_lable)

    #预测
    y_pred1 = svm1.predict(test_data)
    recall=metrics.recall_score(test_data_lable, y_pred1)
    print('召回率: %.2f' % recall)
    f1 = f1_score(test_data_lable, y_pred1)
    print('F1值:%.2f' % f1)
    return recall, f1

'''
  对同一个训练集的单个label训练+交叉验证：
  params:
  X:所有数据特征(包括源域目标域)
  all_lable:所有数据标签(各预测目标)
  label_idx:int 本次需要预测的目标索引
'''
def train_SVM_cross_val(X,all_lable,label_idx):
    print('------------{}--------------'.format(y_columns[label_idx]))
    # train_data_lable = all_train_lable[y_columns[label_idx]]
    # test_data_lable = all_test_lable[y_columns[label_idx]]

    ## 拼接源域目标域, 并取对应任务目标的一列y值
    # X = np.vstack((train_data, test_data))
    # Y = np.vstack((all_train_lable[y_columns[label_idx]], all_test_lable[y_columns[label_idx]]))
    # 取对应任务目标的一列y值
    Y = all_lable[:,label_idx]
    # 查看形状
    print("shape of X:{}\nshape of Y:{}".format(X.shape,Y.shape))
    # print("X:{}\nY:{}".format(X,Y))
    #训练模型
    svm = SVC(kernel='rbf', C=100.0,gamma=100.0, random_state=0)
    # svm1.fit(train_data, train_data_lable)

    # 分层采样+K折交叉
    n_splits=3
    skfolds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    recall=0
    f1=0

    for train_index, test_index in skfolds.split(X, Y):
        clone_clf = clone(svm)
        # X_train_folds = X.iloc[train_index]
        # y_train_folds = (Y.iloc[train_index])
        # X_test_fold = X.iloc[test_index]
        # y_test_fold = (Y.iloc[test_index])
        X_train_folds = X[train_index]
        y_train_folds = (Y[train_index])
        X_test_fold = X[test_index]
        y_test_fold = (Y[test_index])
        #训练, 预测
        clone_clf.fit(X_train_folds, y_train_folds)
        y_pred = clone_clf.predict(X_test_fold)
        recall+=metrics.recall_score(y_test_fold, y_pred)
        f1 += f1_score(y_test_fold, y_pred)

    recall /= n_splits
    f1 /= n_splits
    # y_pred1 = svm.predict(test_data)
    # recall=metrics.recall_score(test_data_lable, y_pred)
    print('召回率: %.2f' % recall)
    # f1 = f1_score(test_data_lable, y_pred)
    print('F1值:%.2f' % f1)
    return recall, f1



# # 对单个label训练时加上随机过采样
# def train_SVM_RandomOverSampling(all_train_lable,all_test_lable,train_data,test_data,label_idx):
#     print('------------{}--------------'.format(y_columns[label_idx]))
#     train_data_lable = all_train_lable[y_columns[label_idx]]
#     test_data_lable = all_test_lable[y_columns[label_idx]]
#     #训练模型
#     svm1 = SVC(kernel='rbf', C=100.0,gamma=100.0, random_state=0)
#     # svm1.fit(train_data, train_data_lable)
#     sampler = RandomOverSampler(random_state=0)
#     # model = make_pipeline(sampler, svm1).fit(train_data, train_data_lable)
#     X_res, y_res = sampler.fit_resample(train_data, train_data_lable)
#     print('Original dataset shape %s' % Counter(train_data_lable))
#     print('Resampled dataset shape %s' % Counter(y_res))
#     svm1.fit(X_res, y_res)
#     #预测
#     # y_pred = model.predict(test_data)
#     y_pred = svm1.predict(test_data)
#     recall=metrics.recall_score(test_data_lable, y_pred)
#     print('召回率: %.2f' % recall)
#     f1 = f1_score(test_data_lable, y_pred)
#     print('F1值:%.2f' % f1)
#     return recall, f1



# 看baseline TCA数据集预测的recall
def all_label_TCA_SVM_train():
    train_data = pd.read_csv('./data/TCA_source_data.csv',index_col=0)
    test_data = pd.read_csv('./data/TCA_target_data.csv',index_col=0)
    # recall = np.zeros((1,6)) 
    # f1 = np.zeros((1,6)) 
    # for i in range(6):
    #     recall[0][i], f1[0][i]=train_SVM(train_data_lable,test_data_lable,train_data,test_data,i)
    recall = np.zeros((6)) 
    f1 = np.zeros((6)) 
    for i in range(6):
        recall[i], f1[i]=train_SVM(train_data_lable,test_data_lable,train_data,test_data,i)
    # 保存预测后结果
    recall = pd.DataFrame(recall)
    f1 = pd.DataFrame(f1)
    recall.to_csv('./data/TCA+SVM_recall.csv')
    f1.to_csv('./data/TCA+SVM_f1.csv')
    print('Performance saved')

# 看baseline TCA+交叉验证
def all_label_TCA_SVM_cross_val_train():
    train_data = pd.read_csv('./data/TCA_source_data.csv',index_col=0)
    test_data = pd.read_csv('./data/TCA_target_data.csv',index_col=0)
    recall = np.zeros(6) 
    f1 = np.zeros(6) 
    for i in range(6):
        recall[i], f1[i]=train_SVM_cross_val(train_data_lable,test_data_lable,train_data,test_data,i)
    # 保存预测后结果
    recall = pd.DataFrame(recall)
    f1 = pd.DataFrame(f1)
    recall.to_csv('./data/TCA+SVM_cross_val_recall.csv')
    f1.to_csv('./data/TCA+SVM_cross_val_f1.csv')
    print('Performance saved')
    

# 新目标TCA+交叉验证
def new_labels_TCA_SVM_cross_val_train(label_num=9):
    # train_data = pd.read_csv('./data/TCA_source_data.csv',index_col=0)
    # test_data = pd.read_csv('./data/TCA_target_data.csv',index_col=0)
    train_data = pd.read_csv('./data/TCA_source_data.csv')
    test_data = pd.read_csv('./data/TCA_target_data.csv')
    # 拼接源域目标域
    data = np.vstack((train_data, test_data))
    # Y = np.vstack((all_train_lable[y_columns[label_idx]], all_test_lable[y_columns[label_idx]]))
    label = np.vstack((train_data_lable, test_data_lable))
    # label_num=9
    recall = np.zeros(label_num) 
    f1 = np.zeros(label_num) 
    for i in range(label_num):
        # recall[i], f1[i]=train_SVM_cross_val(train_data_lable,test_data_lable,train_data,test_data,i)
        recall[i], f1[i]=train_SVM_cross_val(data,label,i)
    # 保存预测后结果
    recall = pd.DataFrame(recall)
    f1 = pd.DataFrame(f1)
    recall.to_csv('./data/{}features_TCA+SVM_cross_val_recall.csv'.format(curtime), index=False)
    f1.to_csv('./data/{}features_TCA+SVM_cross_val_f1.csv'.format(curtime), index=False)
    print('TCA Performance saved')

# 新目标HDA+交叉验证
def new_labels_HDA_SVM_cross_val_train(label_num=9):
    # train_data = pd.read_csv('./data/TCA_source_data.csv',index_col=0)
    # test_data = pd.read_csv('./data/TCA_target_data.csv',index_col=0)
    train_data = pd.read_csv('./data/HDA_s_data.csv')
    test_data = pd.read_csv('./data/HDA_t_data.csv')
    # 拼接源域目标域
    data = np.vstack((train_data, test_data))
    label = np.vstack((train_data_lable, test_data_lable))

    recall = np.zeros(label_num) 
    f1 = np.zeros(label_num) 
    for i in range(label_num):
        # recall[i], f1[i]=train_SVM_cross_val(train_data_lable,test_data_lable,train_data,test_data,i)
        recall[i], f1[i]=train_SVM_cross_val(data,label,i)
    # 保存预测后结果
    recall = pd.DataFrame(recall)
    f1 = pd.DataFrame(f1)
    # recall.to_csv('./data/1.7features_HDA+SVM_cross_val_recall.csv', index=False)
    recall.to_csv('./data/{}features_HDA+SVM_cross_val_recall.csv'.format(curtime), index=False)
    f1.to_csv('./data/{}features_HDA+SVM_cross_val_f1.csv'.format(curtime), index=False)
    # f1.to_csv('./data/1.7features_HDA+SVM_cross_val_f1.csv', index=False)
    print('HDA Performance saved')

# 新目标PCA+交叉验证
def new_labels_PCA_SVM_cross_val_train(label_num=9):
    train_data = pd.read_csv('./data/PCA_data.csv')
    # test_data = pd.read_csv('./data/HDA_t_data.csv')
    data=train_data.values
    label=train_data_lable.values
    # label_num=9
    recall = np.zeros(label_num) 
    f1 = np.zeros(label_num) 
    for i in range(label_num):
        # recall[i], f1[i]=train_SVM_cross_val(train_data_lable,test_data_lable,train_data,test_data,i)
        recall[i], f1[i]=train_SVM_cross_val(data,label,i)
    # 保存预测后结果
    recall = pd.DataFrame(recall)
    f1 = pd.DataFrame(f1)
    
    recall.to_csv('./data/{}features_PCA+SVM_cross_val_recall.csv'.format(curtime), index=False)
    f1.to_csv('./data/{}features_PCA+SVM_cross_val_f1.csv'.format(curtime), index=False)
    print('PCA Performance saved')

# 新目标EF+TCA+交叉验证
def new_labels_EF_TCA_SVM_cross_val_train(label_num=9):
    # train_data = pd.read_csv('./data/TCA_source_data.csv')
    # test_data = pd.read_csv('./data/TCA_target_data.csv')
    # # 拼接源域目标域
    # data = np.vstack((train_data, test_data))
    label = np.vstack((train_data_lable, test_data_lable))
    # label_num=9
    recall = np.zeros(label_num) 
    f1 = np.zeros(label_num) 
    for i in range(label_num):
        data = pd.read_csv('./data/{}_fity{}.csv'.format('TCA_after_EF_data',i))
        data=data.values
        # recall[i], f1[i]=train_SVM_cross_val(train_data_lable,test_data_lable,train_data,test_data,i)
        recall[i], f1[i]=train_SVM_cross_val(data,label,i)
    # 保存预测后结果
    recall = pd.DataFrame(recall)
    f1 = pd.DataFrame(f1)
    recall.to_csv('./data/{}features_EF+TCA+SVM_cross_val_recall.csv'.format(curtime), index=False)
    f1.to_csv('./data/{}features_EF+TCA+SVM_cross_val_f1.csv'.format(curtime), index=False)
    print('TCA Performance saved')
# # -----------------------------------随机过采样
# # TCA加上随机过采样对所有label训练
# def all_label_TCA_RandomOverSampling_SVM_train():
#     train_data = pd.read_csv('./data/TCA_source_data.csv',index_col=0)
#     test_data = pd.read_csv('./data/TCA_target_data.csv',index_col=0)
#     # recall = np.zeros((1,6)) 
#     # f1 = np.zeros((1,6)) 
#     # for i in range(6):
#     #     recall[0][i], f1[0][i]=train_SVM(train_data_lable,test_data_lable,train_data,test_data,i)
#     recall = np.zeros((6)) 
#     f1 = np.zeros((6)) 
#     for i in range(6):
#         recall[i], f1[i]=train_SVM_RandomOverSampling(train_data_lable,test_data_lable,train_data,test_data,i)
#     # 保存预测后结果
#     recall = pd.DataFrame(recall)
#     f1 = pd.DataFrame(f1)
#     recall.to_csv('./data/TCA+RandomOverSampling+SVM_recall.csv')
#     f1.to_csv('./data/TCA+RandomOverSampling+SVM_f1.csv')
#     print('Performance saved')

if __name__ == '__main__':
    # # 看原始TCA数据集预测的recall
    # all_label_TCA_SVM_train()
    # # TCA加上过采样
    # all_label_TCA_RandomOverSampling_SVM_train()
    # # 看原始TCA数据集+交叉验证
    # all_label_TCA_SVM_cross_val_train()
    # 修改源域目标域特征+新目标TCA+SVM+交叉验证
    new_labels_TCA_SVM_cross_val_train(9)
    new_labels_HDA_SVM_cross_val_train(9)
    new_labels_PCA_SVM_cross_val_train(9)
    new_labels_EF_TCA_SVM_cross_val_train(9)
    

    

