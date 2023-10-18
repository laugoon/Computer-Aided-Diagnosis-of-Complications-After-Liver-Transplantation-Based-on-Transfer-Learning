# from turtle import end_fill, shape
from distutils.log import debug
from pyexpat import model
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn import metrics
import  pickle
from sklearn.metrics import f1_score
# from imblearn.pipeline import make_pipeline
# from imblearn.over_sampling import RandomOverSampler,SMOTE,ADASYN
from collections import Counter

#add
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
import time
import shap
from sklearn.decomposition import PCA
from sklearn.linear_model import Lasso
# from sklearn.model_selection import train_test_split
import sys
sys.path.append('./')
from data_process_shap.TCA_shap import TCA,TCA_data_read
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler

#add



#预测标签
# train_data_lable = pd.read_csv('./data/labels.csv')
train_data_lable = pd.read_csv('./data/fill_labels.csv')
test_data_lable = train_data_lable.copy(deep=True)
# # y_columns = ['所有胸腔积液情况']
# y_columns=['胸腔积液','凝血功能指标','转氨酶指标','胆红素指标','急性肺损伤','术后出血','术后感染','胆道并发症','原发性移植肝无功能']
#英文类别名称
y_columns=['Pleural Effusion','High INR','High ALT or AST','High TBIL','Pneumonia','Postoperative Bleeding','Postoperative Infection','Biliary Complications','PNF']

curtime = time.strftime('%Y-%m-%d',time.localtime(time.time()))

# 全局预定义解释器需要变量
# PCA降维
pca=None
# TCA迁移
tca=None
#svm分类器
svm_model=None
#tca在fit_new时用到原先计算变换矩阵的data
source_data=None
target_data=None
# print("train X shape:",train_data.shape)
# print("test X shape:",test_data.shape)
# print("train Y shape:",train_data_lable.shape)
# print("test Y shape:",test_data_lable.shape)

def pca_svm_wrapper(X):
    # PCA降维-测试集
    x_test_r = pca.transform(X)
    # svm分类-测试集
    # y_pred = svm_model.predict(x_test_r)
    y_pred_proba = svm_model.predict_proba(x_test_r)
    return y_pred_proba

def tca_svm_wrapper(X):
    # TCA降维-测试集
    x_test_r = tca.fit_new(source_data, target_data, X)
    # svm分类-测试集
    y_pred_proba = svm_model.predict_proba(x_test_r)
    return y_pred_proba

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

def train_1label_PCA_SVM_wrapper_shap(X,all_lable,label_idx):
    print('------------{}--------------'.format(y_columns[label_idx]))
    # features_num=451
    ## 拼接源域目标域, 并取对应任务目标的一列y值
    # X = np.vstack((train_data, test_data))
    # Y = np.vstack((all_train_lable[y_columns[label_idx]], all_test_lable[y_columns[label_idx]]))
    # 取对应任务目标的一列y值
    Y = all_lable[:,label_idx]
    # 查看形状
    print("shape of X:{}\nshape of Y:{}".format(X.shape,Y.shape))
    # print("X:{}\nY:{}".format(X,Y))
    #训练模型
    svm = SVC(kernel='rbf', C=100.0,gamma=100.0, random_state=0, probability=True)
    # svm1.fit(train_data, train_data_lable)

    # 分层采样+K折交叉
    n_splits=3
    skfolds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    recall=0
    prc=0
    f1=0
    # global_shap_values_1fold=np.empty(shape=(n_splits,features_num))
    # fold_count=0
    # for fold_count, train_index, test_index in enumerate(skfolds.split(X, Y)):
    for train_index, test_index in skfolds.split(X, Y):
        global pca
        global svm_model
        n_components=50
        cluster=100
        model=pca_svm_wrapper
        svm_model = clone(svm)
        # X_train_folds = X.iloc[train_index]
        # y_train_folds = (Y.iloc[train_index])
        # X_test_fold = X.iloc[test_index]
        # y_test_fold = (Y.iloc[test_index])
        X_train_folds = X[train_index]
        y_train_folds = (Y[train_index])
        X_test_fold = X[test_index]
        y_test_fold = (Y[test_index])
        # 降维
        pca = PCA(n_components=n_components)
        x_train_r = pca.fit(X_train_folds).transform(X_train_folds)
        # PCA降维-测试集
        x_test_r = pca.transform(X_test_fold)
        #训练, 预测
        # svm_model.fit(X_train_folds, y_train_folds)
        # y_pred = svm_model.predict(X_test_fold)
        svm_model.fit(x_train_r, y_train_folds)
        y_pred = svm_model.predict(x_test_r)
        recall+=metrics.recall_score(y_test_fold, y_pred)
        prc+=metrics.precision_score(y_test_fold, y_pred)

        f1 += f1_score(y_test_fold, y_pred)
        
        # # KernelExplainer
        # # 0.聚类，为了使计算过程简化，加快速度
        # X_train_summary = shap.kmeans(X_train_folds, cluster)
        # # 1.创建解释器对象
        # # explainer = shap.KernelExplainer(svm_model.predict_proba, x_train, link="logit")
        # explainer = shap.KernelExplainer(model, data=X_train_summary, link="logit")

        # # # 1.创建解释器对象
        # # explainer = shap.KernelExplainer(svm_model.predict_proba, X_train_folds, link="logit")
        # # # 2.计算各个样本的局部shapley值
        # # # shap_values = explainer.shap_values(X_test_fold, nsamples="auto")

        # # debug时nsamples=1, 正式计算100次尝试
        # # shap_values = explainer.shap_values(X_test_fold, nsamples=1)
        # shap_values = explainer.shap_values(X_test_fold, nsamples=100)
        # # 3. 获取该次迭代的全局shap值
        # global_shap_values_1fold[fold_count] = np.abs(shap_values[0]).mean(0) #对第一个小数组(类别0)求均值
        # fold_count=fold_count+1
        # #对k折交叉的各个结果拼接到一起
        # # np.concatenate((shap_values_all,shap_values),axis=0)
        # # shap_values_all=shap_values
        # # 调试
        # # shap.summary_plot(shap_values_all, X_train_folds, plot_type="bar")
        # # KernelExplainer end





    recall /= n_splits
    f1 /= n_splits
    prc /= n_splits
    # y_pred1 = svm.predict(test_data)
    # recall=metrics.recall_score(test_data_lable, y_pred)
    print('召回率: %.2f' % recall)
    # f1 = f1_score(test_data_lable, y_pred)
    print('F1值:%.2f' % f1)

    # # 保存以便查看
    # global_shap_values_1fold = pd.DataFrame(global_shap_values_1fold)
    # global_shap_values_1fold.to_csv('./temp_output/global_shap_values.csv', index=False)
    # # shap.plots.bar(shap_values_all[0])
    # # 全局shap值
    # global_shap_values=global_shap_values_1fold.mean(0)
    # return recall, f1, global_shap_values
    return recall, f1, prc

def train_1label_PCA_SVM_ROS(X,all_lable,label_idx):
    print('------------{}--------------'.format(y_columns[label_idx]))
    features_num=451
    ## 拼接源域目标域, 并取对应任务目标的一列y值
    # X = np.vstack((train_data, test_data))
    # Y = np.vstack((all_train_lable[y_columns[label_idx]], all_test_lable[y_columns[label_idx]]))
    # 取对应任务目标的一列y值
    Y = all_lable[:,label_idx]
    # 查看形状
    print("shape of X:{}\nshape of Y:{}".format(X.shape,Y.shape))
    # print("X:{}\nY:{}".format(X,Y))
    #训练模型
    svm = SVC(kernel='rbf', C=100.0,gamma=100.0, random_state=0, probability=True)
    # svm1.fit(train_data, train_data_lable)

    # 分层采样+K折交叉
    n_splits=3
    skfolds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    recall=0
    prc=0
    f1=0
    global_shap_values_1fold=np.empty(shape=(n_splits,features_num))
    fold_count=0
    # for fold_count, train_index, test_index in enumerate(skfolds.split(X, Y)):
    for train_index, test_index in skfolds.split(X, Y):
        global pca
        global svm_model
        n_components=50
        # cluster=100
        # model=pca_svm_wrapper
        svm_model = clone(svm)
        # X_train_folds = X.iloc[train_index]
        # y_train_folds = (Y.iloc[train_index])
        # X_test_fold = X.iloc[test_index]
        # y_test_fold = (Y.iloc[test_index])
        X_train_folds = X[train_index]
        y_train_folds = (Y[train_index])
        X_test_fold = X[test_index]
        y_test_fold = (Y[test_index])

        # 降维
        pca = PCA(n_components=n_components)
        x_train_r = pca.fit(X_train_folds).transform(X_train_folds)
        # PCA降维-测试集
        x_test_r = pca.transform(X_test_fold)

        # 上采样
        counter=Counter(y_train_folds)
        minor_labels_ratio=counter[1]/(counter[0]+counter[1])
        print("labels_ratio:",minor_labels_ratio)
        if(minor_labels_ratio<0.22):
            sampler = RandomOverSampler(random_state=0)
            X_res, y_res = sampler.fit_resample(x_train_r, y_train_folds)
            print('Original dataset shape %s' % Counter(y_train_folds))
            print('Resampled dataset shape %s' % Counter(y_res))
        else:#没达到阈值则不进行操作
            X_res=x_train_r
            y_res=y_train_folds


        
        #训练, 预测
        # svm_model.fit(X_train_folds, y_train_folds)
        # y_pred = svm_model.predict(X_test_fold)
        svm_model.fit(X_res, y_res)
        y_pred = svm_model.predict(x_test_r)
        recall+=metrics.recall_score(y_test_fold, y_pred)
        prc+=metrics.precision_score(y_test_fold, y_pred)
        f1 += f1_score(y_test_fold, y_pred)
   

    recall /= n_splits
    f1 /= n_splits
    prc /= n_splits
    # y_pred1 = svm.predict(test_data)
    # recall=metrics.recall_score(test_data_lable, y_pred)
    print('准确率: %.2f' % prc)
    print('真值:\n',y_test_fold,'预测值:\n', y_pred)
    print('召回率: %.2f' % recall)
    # f1 = f1_score(test_data_lable, y_pred)
    print('F1值:%.2f' % f1)


    return recall, f1, prc

# LASSO for LR
def train_1label_LASSO_ROS(X,all_lable,feature_names,n_components,label_idx):
    print('------------{}--------------'.format(y_columns[label_idx]))
    features_num=451
    ## 拼接源域目标域, 并取对应任务目标的一列y值
    # X = np.vstack((train_data, test_data))
    # Y = np.vstack((all_train_lable[y_columns[label_idx]], all_test_lable[y_columns[label_idx]]))
    # 取对应任务目标的一列y值
    Y = all_lable[:,label_idx]
    # 查看形状
    print("shape of X:{}\nshape of Y:{}".format(X.shape,Y.shape))
    # print("X:{}\nY:{}".format(X,Y))
    # #训练模型
    # svm = SVC(kernel='rbf', C=100.0,gamma=100.0, random_state=0, probability=True)
    # svm1.fit(train_data, train_data_lable)

    # 分层采样+K折交叉
    n_splits=3
    skfolds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    recall=0
    prc=0
    f1=0
    global_shap_values_1fold=np.empty(shape=(n_splits,features_num))
    top_features=np.empty(shape=(1,n_components))
    fold_count=0
    # for fold_count, train_index, test_index in enumerate(skfolds.split(X, Y)):
    for train_index, test_index in skfolds.split(X, Y):
        # global pca
        # global svm_model
        # n_components=50
        # cluster=100
        # model=pca_svm_wrapper
        # svm_model = clone(svm)

        # X_train_folds = X.iloc[train_index]
        # y_train_folds = (Y.iloc[train_index])
        # X_test_fold = X.iloc[test_index]
        # y_test_fold = (Y.iloc[test_index])
        X_train_folds = X[train_index]
        y_train_folds = (Y[train_index])
        X_test_fold = X[test_index]
        y_test_fold = (Y[test_index])

        # # 降维LASSO
        # lasso = Lasso(alpha=10**(-3))
        # model_lasso = lasso.fit(X_train_folds, y_train_folds)
        # coef = pd.Series(model_lasso.coef_,index=feature_names)
        # print(coef[coef != 0].abs().sort_values(ascending = False))
        # # PCA
        # pca = PCA(n_components=n_components)
        # x_train_r = pca.fit(X_train_folds).transform(X_train_folds)
        # PCA降维-测试集
        # x_test_r = pca.transform(X_test_fold)

       


        
        # # 训练, 预测
        # svm_model.fit(X_train_folds, y_train_folds)
        # y_pred = svm_model.predict(X_test_fold)
        # svm_model.fit(X_res, y_res)
        # y_pred = svm_model.predict(x_test_r)
        # model_lasso.fit(X_train_folds, y_train_folds)
        # y_pred = model_lasso.predict(X_test_fold)
        
        # 对训练集和测试集进行标准化处理
        scaler = StandardScaler()
        X_train_folds = scaler.fit_transform(X_train_folds)
        X_test_fold = scaler.transform(X_test_fold)

        # 上采样
        counter=Counter(y_train_folds)
        minor_labels_ratio=counter[1]/(counter[0]+counter[1])
        print("labels_ratio:",minor_labels_ratio)
        if(minor_labels_ratio<0.22):
            sampler = RandomOverSampler(random_state=0)
            X_res, y_res = sampler.fit_resample(X_train_folds, y_train_folds)
            print('Original dataset shape %s' % Counter(y_train_folds))
            print('Resampled dataset shape %s' % Counter(y_res))
        else:#没达到阈值则不进行操作
            X_res=X_train_folds
            y_res=y_train_folds

        # # 使用LASSO进行特征选择(未上采样)
        # logistic = LogisticRegression(penalty='l1', solver='liblinear')
        # selector = SelectFromModel(logistic)
        # selector.fit(X_train_folds, y_train_folds)

        # # 获取选择后的特征(未上采样)
        # X_train_selected = selector.transform(X_train_folds)
        # X_test_selected = selector.transform(X_test_fold)

        # # 训练逻辑回归模型(未上采样)
        # logistic_selected = LogisticRegression()
        # logistic_selected.fit(X_train_selected, y_train_folds)

        # 使用LASSO进行特征选择(上采样)
        logistic = LogisticRegression(penalty='l1', solver='liblinear')
        selector = SelectFromModel(logistic,max_features=n_components)
        selector.fit(X_res, y_res)
        top_features=selector.get_feature_names_out(feature_names).flatten()
        # 获取选择后的特征(上采样)
        X_train_selected = selector.transform(X_res)
        X_test_selected = selector.transform(X_test_fold)

        # 训练逻辑回归模型(上采样)
        logistic_selected = LogisticRegression()
        logistic_selected.fit(X_train_selected, y_res)
        
        # 预测测试集的标签
        y_pred = logistic_selected.predict(X_test_selected)

        # 计算模型性能指标
        # accuracy = accuracy_score(y_test_fold, y_pred)
        # precision = precision_score(y_test_fold, y_pred)
        # recall = recall_score(y_test_fold, y_pred)
        # f1 = f1_score(y_test_fold, y_pred)


        recall+=metrics.recall_score(y_test_fold, y_pred)
        prc+=metrics.precision_score(y_test_fold, y_pred)
        f1 += f1_score(y_test_fold, y_pred)
   

    recall /= n_splits
    f1 /= n_splits
    prc /= n_splits
    # y_pred1 = svm.predict(test_data)
    # recall=metrics.recall_score(test_data_lable, y_pred)
    print('准确率: %.2f' % prc)
    print('真值:\n',y_test_fold,'预测值:\n', y_pred)
    print('召回率: %.2f' % recall)
    # f1 = f1_score(test_data_lable, y_pred)
    print('F1值:%.2f' % f1)
    print("LASSO top features:",top_features)


    return recall, f1, prc,top_features

# LASSO for LR
def train_1label_LASSO_SVM_ROS(X,all_lable,feature_names,n_components,label_idx):
    print('------------{}--------------'.format(y_columns[label_idx]))
    features_num=451
    ## 拼接源域目标域, 并取对应任务目标的一列y值
    # X = np.vstack((train_data, test_data))
    # Y = np.vstack((all_train_lable[y_columns[label_idx]], all_test_lable[y_columns[label_idx]]))
    # 取对应任务目标的一列y值
    Y = all_lable[:,label_idx]
    # 查看形状
    print("shape of X:{}\nshape of Y:{}".format(X.shape,Y.shape))
    # print("X:{}\nY:{}".format(X,Y))
    # #训练模型
    svm = SVC(kernel='rbf', C=100.0,gamma=100.0, random_state=0, probability=True)
    # svm1.fit(train_data, train_data_lable)

    # 分层采样+K折交叉
    n_splits=3
    skfolds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    recall=0
    prc=0
    f1=0
    global_shap_values_1fold=np.empty(shape=(n_splits,features_num))
    top_features=np.empty(shape=(1,n_components))
    fold_count=0
    # for fold_count, train_index, test_index in enumerate(skfolds.split(X, Y)):
    for train_index, test_index in skfolds.split(X, Y):
        # global pca
        # global svm_model
        # n_components=50
        # cluster=100
        # model=pca_svm_wrapper
        svm_model = clone(svm)

        # X_train_folds = X.iloc[train_index]
        # y_train_folds = (Y.iloc[train_index])
        # X_test_fold = X.iloc[test_index]
        # y_test_fold = (Y.iloc[test_index])
        X_train_folds = X[train_index]
        y_train_folds = (Y[train_index])
        X_test_fold = X[test_index]
        y_test_fold = (Y[test_index])

        # # 降维LASSO
        # lasso = Lasso(alpha=10**(-3))
        # model_lasso = lasso.fit(X_train_folds, y_train_folds)
        # coef = pd.Series(model_lasso.coef_,index=feature_names)
        # print(coef[coef != 0].abs().sort_values(ascending = False))
        # # PCA
        # pca = PCA(n_components=n_components)
        # x_train_r = pca.fit(X_train_folds).transform(X_train_folds)
        # PCA降维-测试集
        # x_test_r = pca.transform(X_test_fold)
        
        # 对训练集和测试集进行标准化处理
        scaler = StandardScaler()
        X_train_folds = scaler.fit_transform(X_train_folds)
        X_test_fold = scaler.transform(X_test_fold)

        # 上采样
        counter=Counter(y_train_folds)
        minor_labels_ratio=counter[1]/(counter[0]+counter[1])
        print("labels_ratio:",minor_labels_ratio)
        if(minor_labels_ratio<0.22):
            sampler = RandomOverSampler(random_state=0)
            X_res, y_res = sampler.fit_resample(X_train_folds, y_train_folds)
            print('Original dataset shape %s' % Counter(y_train_folds))
            print('Resampled dataset shape %s' % Counter(y_res))
        else:#没达到阈值则不进行操作
            X_res=X_train_folds
            y_res=y_train_folds

        # # 使用LASSO进行特征选择(未上采样)
        # logistic = LogisticRegression(penalty='l1', solver='liblinear')
        # selector = SelectFromModel(logistic)
        # selector.fit(X_train_folds, y_train_folds)

        # # 获取选择后的特征(未上采样)
        # X_train_selected = selector.transform(X_train_folds)
        # X_test_selected = selector.transform(X_test_fold)

        # # 训练逻辑回归模型(未上采样)
        # logistic_selected = LogisticRegression()
        # logistic_selected.fit(X_train_selected, y_train_folds)

        # 使用LASSO进行特征选择(上采样)
        logistic = LogisticRegression(penalty='l1', solver='liblinear')
        selector = SelectFromModel(logistic,max_features=n_components)
        selector.fit(X_res, y_res)
        top_features=selector.get_feature_names_out(feature_names).flatten()
        # 获取选择后的特征(上采样)
        X_train_selected = selector.transform(X_res)
        X_test_selected = selector.transform(X_test_fold)

        # SVM训练, 预测
        svm_model.fit(X_train_selected, y_res)
        y_pred = svm_model.predict(X_test_selected)

        # # 训练逻辑回归模型(上采样)
        # logistic_selected = LogisticRegression()
        # logistic_selected.fit(X_train_selected, y_res)
        
        # # 预测测试集的标签
        # y_pred = logistic_selected.predict(X_test_selected)

        # 计算模型性能指标
        # accuracy = accuracy_score(y_test_fold, y_pred)
        # precision = precision_score(y_test_fold, y_pred)
        # recall = recall_score(y_test_fold, y_pred)
        # f1 = f1_score(y_test_fold, y_pred)


        recall+=metrics.recall_score(y_test_fold, y_pred)
        prc+=metrics.precision_score(y_test_fold, y_pred)
        f1 += f1_score(y_test_fold, y_pred)
   

    recall /= n_splits
    f1 /= n_splits
    prc /= n_splits
    # y_pred1 = svm.predict(test_data)
    # recall=metrics.recall_score(test_data_lable, y_pred)
    print('准确率: %.2f' % prc)
    print('真值:\n',y_test_fold,'预测值:\n', y_pred)
    print('召回率: %.2f' % recall)
    # f1 = f1_score(test_data_lable, y_pred)
    print('F1值:%.2f' % f1)
    print("LASSO top features:",top_features)


    return recall, f1, prc,top_features

def train_1label_HDA_SVM_ROS(X,all_lable,label_idx):
    print('------------{}--------------'.format(y_columns[label_idx]))
    # features_num=451
    ## 拼接源域目标域, 并取对应任务目标的一列y值
    # X = np.vstack((train_data, test_data))
    # Y = np.vstack((all_train_lable[y_columns[label_idx]], all_test_lable[y_columns[label_idx]]))
    # 取对应任务目标的一列y值
    Y = all_lable[:,label_idx]
    # 查看形状
    print("shape of X:{}\nshape of Y:{}".format(X.shape,Y.shape))
    # print("X:{}\nY:{}".format(X,Y))
    #训练模型
    svm = SVC(kernel='rbf', C=100.0,gamma=100.0, random_state=0, probability=True)
    # svm1.fit(train_data, train_data_lable)

    # 分层采样+K折交叉
    n_splits=3
    skfolds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    recall=0
    prc=0
    f1=0
    # global_shap_values_1fold=np.empty(shape=(n_splits,features_num))
    fold_count=0
    # for fold_count, train_index, test_index in enumerate(skfolds.split(X, Y)):
    for train_index, test_index in skfolds.split(X, Y):
        # global pca
        global svm_model
        n_components=50
        # cluster=100
        # model=pca_svm_wrapper
        svm_model = clone(svm)
        # X_train_folds = X.iloc[train_index]
        # y_train_folds = (Y.iloc[train_index])
        # X_test_fold = X.iloc[test_index]
        # y_test_fold = (Y.iloc[test_index])
        X_train_folds = X[train_index]
        y_train_folds = (Y[train_index])
        X_test_fold = X[test_index]
        y_test_fold = (Y[test_index])

        # # 降维
        # pca = PCA(n_components=n_components)
        # x_train_r = pca.fit(X_train_folds).transform(X_train_folds)
        # # PCA降维-测试集
        # x_test_r = pca.transform(X_test_fold)

        # 上采样
        counter=Counter(y_train_folds)
        minor_labels_ratio=counter[1]/(counter[0]+counter[1])
        print("labels_ratio:",minor_labels_ratio)
        if(minor_labels_ratio<0.22):
            sampler = RandomOverSampler(random_state=0)
            X_res, y_res = sampler.fit_resample(X_train_folds, y_train_folds)
            print('Original dataset shape %s' % Counter(y_train_folds))
            print('Resampled dataset shape %s' % Counter(y_res))
        else:#没达到阈值则不进行操作
            X_res=X_train_folds
            y_res=y_train_folds


        
        #训练, 预测
        # svm_model.fit(X_train_folds, y_train_folds)
        # y_pred = svm_model.predict(X_test_fold)
        svm_model.fit(X_res, y_res)
        y_pred = svm_model.predict(X_test_fold)
        recall+=metrics.recall_score(y_test_fold, y_pred)
        prc+=metrics.precision_score(y_test_fold, y_pred)
        f1 += f1_score(y_test_fold, y_pred)
   

    recall /= n_splits
    f1 /= n_splits
    prc /= n_splits
    # y_pred1 = svm.predict(test_data)
    # recall=metrics.recall_score(test_data_lable, y_pred)
    print('召回率: %.2f' % recall)
    # f1 = f1_score(test_data_lable, y_pred)
    print('F1值:%.2f' % f1)


    return recall, f1, prc

# 下采样
def train_1label_TCA_SVM_RUS_unshap(Xs,all_ys,Xt,all_yt,label_idx):
    print('------------{}--------------'.format(y_columns[label_idx]))
    features_num=Xs.shape[1]
    # 查看形状
    print("shape of X:{}\nshape of Y:{}".format(Xs.shape,all_ys.shape))
    # 取对应任务目标的一列y值
    Ys = all_ys[:,label_idx]
    Yt = all_yt[:,label_idx]
    
    #训练模型
    svm = SVC(kernel='rbf', C=100.0,gamma=100.0, random_state=0, probability=True)

    # 分层采样+K折交叉
    n_splits=3
    skfolds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)

    recall=0
    f1=0
    prc=0
    global_shap_values_1fold=np.empty(shape=(n_splits,features_num))
    fold_count=0
    # 分折的时候Xt, Yt分, Xs留着跟Xt_train拼接
    for train_index, test_index in skfolds.split(Xt, Yt):
    # xt_train, xt_test, yt_train, yt_test 
        global tca
        global svm_model
        global source_data
        global target_data
        n_components=50
        cluster=100
        model=tca_svm_wrapper
        svm_model = clone(svm)

        Xt_train_folds = Xt[train_index]
        yt_train_folds = Yt[train_index]
        Xt_test_fold = Xt[test_index]
        yt_test_fold = Yt[test_index]
        # 全局变量管理
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
        
        # 下采样
        # sampler = RandomOverSampler(random_state=0)
        sampler = RandomUnderSampler(random_state=0)
        X_res, y_res = sampler.fit_resample(x_train_r, y_train)
        print('Original dataset shape %s' % Counter(y_train))
        print('Resampled dataset shape %s' % Counter(y_res))

        #训练, 预测
        # svm_model.fit(x_train_r, y_train)
        svm_model.fit(X_res, y_res)#下采样
        y_pred = svm_model.predict(Xt_test_new)
        recall+=metrics.recall_score(yt_test_fold, y_pred)
        f1 += f1_score(yt_test_fold, y_pred)
        prc+=metrics.precision_score(yt_test_fold, y_pred)
        
        # # KernelExplainer
        # # 0.聚类，为了使计算过程简化，加快速度
        # X_train_summary = shap.kmeans(x_train, cluster)
        # # 1.创建解释器对象
        # # explainer = shap.KernelExplainer(svm_model.predict_proba, x_train, link="logit")
        # explainer = shap.KernelExplainer(model, data=X_train_summary, link="logit")

        # # # 1.创建解释器对象
        # # explainer = shap.KernelExplainer(svm_model.predict_proba, X_train_folds, link="logit")
        # # # 2.计算各个样本的局部shapley值
        # # # shap_values = explainer.shap_values(X_test_fold, nsamples="auto")

        # # debug时nsamples=1, 正式计算100次尝试
        # shap_values = explainer.shap_values(Xt_test_fold, nsamples=100)
        # # shap_values = explainer.shap_values(Xt_test_fold, nsamples=100)
        # # 3. 获取该次迭代的全局shap值
        # global_shap_values_1fold[fold_count] = np.abs(shap_values[0]).mean(0) #对第一个小数组(类别0)求均值
        # fold_count=fold_count+1
        # #对k折交叉的各个结果拼接到一起
        # # np.concatenate((shap_values_all,shap_values),axis=0)
        # # shap_values_all=shap_values
        # # 调试
        # # shap.summary_plot(shap_values_all, X_train_folds, plot_type="bar")
        # # KernelExplainer end

    recall /= n_splits
    f1 /= n_splits
    prc /= n_splits
    # y_pred1 = svm.predict(test_data)
    # recall=metrics.recall_score(test_data_lable, y_pred)
    print('召回率: %.2f' % recall)
    # f1 = f1_score(test_data_lable, y_pred)
    print('F1值:%.2f' % f1)

    # # 保存以便查看
    # global_shap_values_1fold = pd.DataFrame(global_shap_values_1fold)
    # global_shap_values_1fold.to_csv('./temp_output/global_shap_values.csv', index=False)
    # # shap.plots.bar(shap_values_all[0])
    # 全局shap值
    global_shap_values=global_shap_values_1fold.mean(0)
    # return recall, f1, global_shap_values
    return recall, f1, prc

'''测试集上采样'''
def train_1label_TCA_SVM_ROS_testdata(Xs,all_ys,Xt,all_yt,label_idx):
    print('------------{}--------------'.format(y_columns[label_idx]))
    features_num=Xs.shape[1]
    # 查看形状
    print("shape of X:{}\nshape of Y:{}".format(Xs.shape,all_ys.shape))
    # 取对应任务目标的一列y值
    Ys = all_ys[:,label_idx]
    Yt = all_yt[:,label_idx]
    
    #训练模型
    svm = SVC(kernel='rbf', C=100.0,gamma=100.0, random_state=0, probability=True)

    # 分层采样+K折交叉
    n_splits=3
    skfolds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)

    recall=0
    f1=0
    prc=0
    global_shap_values_1fold=np.empty(shape=(n_splits,features_num))
    fold_count=0
    # 分折的时候Xt, Yt分, Xs留着跟Xt_train拼接
    for train_index, test_index in skfolds.split(Xt, Yt):
    # xt_train, xt_test, yt_train, yt_test 
        global tca
        global svm_model
        global source_data
        global target_data
        n_components=50
        cluster=100
        model=tca_svm_wrapper
        svm_model = clone(svm)

        Xt_train_folds = Xt[train_index]
        yt_train_folds = Yt[train_index]
        Xt_test_fold = Xt[test_index]
        yt_test_fold = Yt[test_index]
        # 全局变量管理
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
        
        # 上采样
        sampler = RandomOverSampler(random_state=0)
        X_res, y_res = sampler.fit_resample(x_train_r, y_train)
        # print('Original dataset shape %s' % Counter(y_train))
        # print('Resampled dataset shape %s' % Counter(y_res))
        # 测试集上采样
        Xt_test_res, y_test_res = sampler.fit_resample(Xt_test_new, yt_test_fold)
        print('Original dataset shape %s' % Counter(yt_test_fold))
        print('Resampled dataset shape %s' % Counter(y_test_res))

        #训练, 预测
        # svm_model.fit(x_train_r, y_train)
        # y_pred = svm_model.predict(Xt_test_new)
        # recall+=metrics.recall_score(yt_test_fold, y_pred)
        # f1 += f1_score(yt_test_fold, y_pred)
        # prc+=metrics.precision_score(yt_test_fold, y_pred)
        #上采样
        svm_model.fit(X_res, y_res)
        y_pred = svm_model.predict(Xt_test_res)
        recall+=metrics.recall_score(y_test_res, y_pred)
        f1 += f1_score(y_test_res, y_pred)
        prc+=metrics.precision_score(y_test_res, y_pred)



    recall /= n_splits
    f1 /= n_splits
    prc /= n_splits
    # y_pred1 = svm.predict(test_data)
    # recall=metrics.recall_score(test_data_lable, y_pred)
    print('召回率: %.2f' % recall)
    # f1 = f1_score(test_data_lable, y_pred)
    print('F1值:%.2f' % f1)


    return recall, f1, prc



'''标准数据集划分-上采样'''
def train_1label_TCA_SVM_ROS_shap(Xs,all_ys,Xt,all_yt,label_idx):
    print('------------{}--------------'.format(y_columns[label_idx]))
    features_num=Xs.shape[1]
    # 查看形状
    print("shape of X:{}\nshape of Y:{}".format(Xs.shape,all_ys.shape))
    # 取对应任务目标的一列y值
    Ys = all_ys[:,label_idx]
    Yt = all_yt[:,label_idx]
    
    #训练模型
    svm = SVC(kernel='rbf', C=100.0,gamma=100.0, random_state=0, probability=True)

    # 分层采样+K折交叉
    n_splits=3
    skfolds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)

    recall=0
    f1=0
    prc=0
    global_shap_values_1fold=np.empty(shape=(n_splits,features_num))
    fold_count=0
    # 分折的时候Xt, Yt分, Xs留着跟Xt_train拼接
    for train_index, test_index in skfolds.split(Xt, Yt):
    # xt_train, xt_test, yt_train, yt_test 
        global tca
        global svm_model
        global source_data
        global target_data
        n_components=50
        cluster=100
        model=tca_svm_wrapper
        svm_model = clone(svm)

        Xt_train_folds = Xt[train_index]
        yt_train_folds = Yt[train_index]
        Xt_test_fold = Xt[test_index]
        yt_test_fold = Yt[test_index]
        # 全局变量管理
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
        # svm_model.fit(x_train_r, y_train)
        svm_model.fit(X_res, y_res)#上采样后训练模型
        y_pred = svm_model.predict(Xt_test_new)
        recall+=metrics.recall_score(yt_test_fold, y_pred)
        f1 += f1_score(yt_test_fold, y_pred)
        prc+=metrics.precision_score(yt_test_fold, y_pred)
        
        # KernelExplainer
        # 0.聚类，为了使计算过程简化，加快速度
        X_train_summary = shap.kmeans(x_train, cluster)
        # 1.创建解释器对象
        # explainer = shap.KernelExplainer(svm_model.predict_proba, x_train, link="logit")
        explainer = shap.KernelExplainer(model, data=X_train_summary, link="logit")

        # # 1.创建解释器对象
        # explainer = shap.KernelExplainer(svm_model.predict_proba, X_train_folds, link="logit")
        # # 2.计算各个样本的局部shapley值
        # # shap_values = explainer.shap_values(X_test_fold, nsamples="auto")

        # debug时nsamples=1, 正式计算100次尝试
        # shap_values = explainer.shap_values(Xt_test_fold, nsamples=100)
        shap_values = explainer.shap_values(Xt_test_fold, nsamples=1)
        # 3. 获取该次迭代的全局shap值
        global_shap_values_1fold[fold_count] = np.abs(shap_values[0]).mean(0) #对第一个小数组(类别0)求均值
        fold_count=fold_count+1

        # # KernelExplainer end

    recall /= n_splits
    f1 /= n_splits
    prc /= n_splits
    print('F1值:%.2f' % f1)

    # # 保存以便查看
    # global_shap_values_1fold = pd.DataFrame(global_shap_values_1fold)
    # global_shap_values_1fold.to_csv('./temp_output/global_shap_values.csv', index=False)
    # # shap.plots.bar(shap_values_all[0])
    # 全局shap值
    global_shap_values=global_shap_values_1fold.mean(0)
    # return recall, f1, global_shap_values
    return recall, f1, prc, global_shap_values

''''上采样+无shap'''
def train_1label_TCA_SVM_ROS_unshap(Xs,all_ys,Xt,all_yt,label_idx):
    print('------------{}--------------'.format(y_columns[label_idx]))
    features_num=Xs.shape[1]
    # 查看形状
    print("shape of X:{}\nshape of Y:{}".format(Xs.shape,all_ys.shape))
    # 取对应任务目标的一列y值
    Ys = all_ys[:,label_idx]
    Yt = all_yt[:,label_idx]
    
    #训练模型
    svm = SVC(kernel='rbf', C=100.0,gamma=100.0, random_state=0, probability=True)
    # 分层采样+K折交叉
    n_splits=3
    skfolds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)

    recall=0
    f1=0
    prc=0
    # global_shap_values_1fold=np.empty(shape=(n_splits,features_num))
    fold_count=0
    # 分折的时候Xt, Yt分, Xs留着跟Xt_train拼接
    for train_index, test_index in skfolds.split(Xt, Yt):
    # xt_train, xt_test, yt_train, yt_test 
        global tca
        global svm_model
        global source_data
        global target_data
        n_components=50
        cluster=100
        model=tca_svm_wrapper
        svm_model = clone(svm)

        Xt_train_folds = Xt[train_index]
        yt_train_folds = Yt[train_index]
        Xt_test_fold = Xt[test_index]
        yt_test_fold = Yt[test_index]
        # 全局变量管理
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
        # svm_model.fit(x_train_r, y_train)
        svm_model.fit(X_res, y_res)#上采样后训练模型
        y_pred = svm_model.predict(Xt_test_new)
        recall+=metrics.recall_score(yt_test_fold, y_pred)
        f1 += f1_score(yt_test_fold, y_pred)
        prc+=metrics.precision_score(yt_test_fold, y_pred)


        ##################ROC#########################
        proba = svm_model.predict_proba(Xt_test_new)
        
        FPR, tpr, thresholds = roc_curve(y_true=yt_test_fold,  # 真实标签是
                                    y_score=svm_model.decision_function(Xt_test_new),  # 置信度，也可以是概率值
                                    pos_label=1)  # 正样本标签是1，也就是少数类。
        # print(FPR.shape)  # 得到了每一个阈值下的 FPR 数组
        # print(recall_under_thresholds.shape)  # 得到了每一个阈值下的 Recall 数组
        # print(thresholds.shape)  # 如果y_score使用的是距离，那么thresholds也是距离；如果y_score使用的是概率，那么thresholds也是概率

        plt.figure()
        plt.plot(FPR, tpr, c='blue', label='ROC curve')  # ROC 曲线
        plt.title('{} ROC'.format(y_columns[label_idx]))  # 设置标题
        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.legend(loc='lower right')
        # plt.show()
        plt.savefig("./fig/ROC/{}_ROC{}.png".format(y_columns[label_idx],fold_count))
        # 计算AUC面积

        auc = roc_auc_score(y_true=yt_test_fold,  # 真实标签是
                            y_score=svm_model.decision_function(Xt_test_new))  # 置信度，也可以是概率值
        print("AUC:",auc)
        #################保存模型################
        #保存模型
        model_file = './save_model/{}_TCA+SVM{}.model'.format(y_columns[label_idx],fold_count)
        with open(model_file, 'wb') as fout:
            pickle.dump(svm_model, fout)
        # #读取模型
        # model_file = './save_model/SVM-并发症I.model'
        # svm1=pickle.load(open(model_file, 'rb'))
        # print('svm1:',svm1,'\ntest_data shape:',test_data.shape)


        # ################# KernelExplainer ###########################
        # # 0.聚类，为了使计算过程简化，加快速度
        # X_train_summary = shap.kmeans(x_train, cluster)
        # # 1.创建解释器对象
        # # explainer = shap.KernelExplainer(svm_model.predict_proba, x_train, link="logit")
        # explainer = shap.KernelExplainer(model, data=X_train_summary, link="logit")

        # # # 1.创建解释器对象
        # # explainer = shap.KernelExplainer(svm_model.predict_proba, X_train_folds, link="logit")
        # # # 2.计算各个样本的局部shapley值
        # # # shap_values = explainer.shap_values(X_test_fold, nsamples="auto")

        # # debug时nsamples=1, 正式计算100次尝试
        # shap_values = explainer.shap_values(Xt_test_fold, nsamples=100)
        # # shap_values = explainer.shap_values(Xt_test_fold, nsamples=1)
        # # 3. 获取该次迭代的全局shap值
        # global_shap_values_1fold[fold_count] = np.abs(shap_values[0]).mean(0) #对第一个小数组(类别0)求均值
        # fold_count=fold_count+1

        # # # KernelExplainer end
        fold_count=(fold_count+1)%3

    recall /= n_splits
    f1 /= n_splits
    prc /= n_splits
    print('F1值:%.2f' % f1)

    # # 保存以便查看
    # global_shap_values_1fold = pd.DataFrame(global_shap_values_1fold)
    # global_shap_values_1fold.to_csv('./temp_output/global_shap_values.csv', index=False)
    # # shap.plots.bar(shap_values_all[0])
    # 全局shap值
    # global_shap_values=global_shap_values_1fold.mean(0)
    # return recall, f1, prc, global_shap_values
    
    return recall, f1, prc

'''原数据集划分'''
# def train_1label_TCA_SVM_wrapper_shap(X,X_r,all_lable,label_idx):
#     print('------------{}--------------'.format(y_columns[label_idx]))
#     features_num=X.shape[1]
#     # 查看形状
#     print("shape of X:{}\nshape of Y:{}".format(X.shape,all_lable.shape))
#     # 取对应任务目标的一列y值
#     Y = all_lable[:,label_idx]
    
#     #训练模型
#     svm = SVC(kernel='rbf', C=100.0,gamma=100.0, random_state=0, probability=True)

#     # 分层采样+K折交叉
#     n_splits=3
#     skfolds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

#     recall=0
#     f1=0
#     global_shap_values_1fold=np.empty(shape=(n_splits,features_num))
#     fold_count=0
#     # 分折的时候用全部数据集分, 不过使用的是未迁移前的
#     for train_index, test_index in skfolds.split(X, Y):
#     # xt_train, xt_test, yt_train, yt_test 
#         global svm_model
#         # global source_data
#         # global target_data
#         cluster=100
#         model=tca_svm_wrapper
#         svm_model = clone(svm)

#         X_train_folds = X[train_index]
#         y_train_folds = Y[train_index]
#         X_test_fold = X[test_index]
#         y_test_fold = Y[test_index]
#         # 降维后
#         X_r_train_folds = X_r[train_index]
        
#         #TCA测试集
#         Xt_test_new = tca.fit_new(source_data, target_data, X_test_fold)

#         # # 拼接源域目标域的训练集
#         # x_train_r = np.vstack((Xs_new, Xt_train_new))
#         # y_train = np.hstack((Ys, yt_train_folds))
#         # # 背景数据集是迁移之前的训练集(源域+目标域的训练部分)组合: X
#         # x_train = np.vstack((source_data, target_data))
        

#         #训练, 预测
#         svm_model.fit(X_r_train_folds, y_train_folds)
#         y_pred = svm_model.predict(Xt_test_new)
#         recall+=metrics.recall_score(y_test_fold, y_pred)
#         f1 += f1_score(y_test_fold, y_pred)
        
#         # KernelExplainer
#         # 0.聚类，为了使计算过程简化，加快速度
#         X_train_summary = shap.kmeans(X_train_folds, cluster)
#         # 1.创建解释器对象
#         # explainer = shap.KernelExplainer(svm_model.predict_proba, x_train, link="logit")
#         explainer = shap.KernelExplainer(model, data=X_train_summary, link="logit")

#         # # 1.创建解释器对象
#         # explainer = shap.KernelExplainer(svm_model.predict_proba, X_train_folds, link="logit")
#         # # 2.计算各个样本的局部shapley值
#         # # shap_values = explainer.shap_values(X_test_fold, nsamples="auto")

#         # debug时nsamples=1, 正式计算100次尝试
#         shap_values = explainer.shap_values(X_test_fold, nsamples=100)
#         # shap_values = explainer.shap_values(Xt_test_fold, nsamples=100)
#         # 3. 获取该次迭代的全局shap值
#         global_shap_values_1fold[fold_count] = np.abs(shap_values[0]).mean(0) #对第一个小数组(类别0)求均值
#         fold_count=fold_count+1
#         #对k折交叉的各个结果拼接到一起
#         # np.concatenate((shap_values_all,shap_values),axis=0)
#         # shap_values_all=shap_values
#         # 调试
#         # shap.summary_plot(shap_values_all, X_train_folds, plot_type="bar")
#         # KernelExplainer end

#     recall /= n_splits
#     f1 /= n_splits
#     # y_pred1 = svm.predict(test_data)
#     # recall=metrics.recall_score(test_data_lable, y_pred)
#     print('召回率: %.2f' % recall)
#     # f1 = f1_score(test_data_lable, y_pred)
#     print('F1值:%.2f' % f1)

#     # # 保存以便查看
#     # global_shap_values_1fold = pd.DataFrame(global_shap_values_1fold)
#     # global_shap_values_1fold.to_csv('./temp_output/global_shap_values.csv', index=False)
#     # # shap.plots.bar(shap_values_all[0])
#     # 全局shap值
#     global_shap_values=global_shap_values_1fold.mean(0)
#     return recall, f1, global_shap_values


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
    

# 上采样
def TCA_SVM_cross_val_ROS_train(label_num=9):
    # train_data = pd.read_csv('./data/TCA_source_data.csv')
    # test_data = pd.read_csv('./data/TCA_target_data.csv')
    # source_data_lable = pd.read_csv('./data/fill_labels.csv')
    # target_data_lable = source_data_lable.copy(deep=True)
    xs,ys, xt,yt=TCA_data_read()
    # 查看形状
    print("shape of X:{}\nshape of Y:{}".format(xs.shape,ys.shape))
    

    '''标准数据集划分'''
    feature_num=xs.shape[1]
    recall = np.zeros(label_num) 
    f1 = np.zeros(label_num) 
    prc= np.zeros(label_num)
    global_shap=np.empty(shape=(label_num,feature_num))
    for i in range(label_num):
        # recall[i], f1[i], global_shap[i]=train_1label_TCA_SVM_wrapper_shap(xs,ys, xt,yt,i)

        # 无shap+有precision
        recall[i], f1[i], prc[i]=train_1label_TCA_SVM_ROS_shap(xs,ys, xt,yt,i)
    '''标准数据集划分end'''	
    # 保存预测后结果
    recall = pd.DataFrame(recall)
    f1 = pd.DataFrame(f1)
    prc = pd.DataFrame(prc)
    # global_shap = pd.DataFrame(global_shap)
    performance = pd.concat([f1, recall, prc], axis=1, sort=False)
    # 设置表头
    performance.columns = ['F1','Recall','Precision']
    performance.to_csv('./data/imbalance_experiment/{}TCA+SVM_ROS_performance.csv'.format(curtime), index=False)

    # recall.to_csv('./data/imbalance_experiment/{}TCA+SVM_Oversampling_recall.csv'.format(curtime), index=False)
    # f1.to_csv('./data/imbalance_experiment/{}TCA+SVM_Oversampling_f1.csv'.format(curtime), index=False)
    # prc.to_csv('./data/imbalance_experiment/{}TCA+SVM_Oversampling_prc.csv'.format(curtime), index=False)
    
    # global_shap.to_csv('./data/wrapper_experiment/{}TCA+SVM_wrapper_shap.csv'.format(curtime), index=False)
    print('TCA Performance saved')

# 上采样
def TCA_SVM_cross_val_ROS_shap_train(label_num=9):
    # train_data = pd.read_csv('./data/TCA_source_data.csv')
    # test_data = pd.read_csv('./data/TCA_target_data.csv')
    # source_data_lable = pd.read_csv('./data/fill_labels.csv')
    # target_data_lable = source_data_lable.copy(deep=True)
    xs,ys, xt,yt=TCA_data_read()
    # 查看形状
    print("shape of X:{}\nshape of Y:{}".format(xs.shape,ys.shape))
    

    '''标准数据集划分'''
    feature_num=xs.shape[1]
    recall = np.zeros(label_num) 
    f1 = np.zeros(label_num) 
    prc= np.zeros(label_num)
    global_shap=np.empty(shape=(label_num,feature_num))
    for i in range(label_num):
    # for i in range(5):
        # # 无shap+有precision
        # recall[i], f1[i], prc[i]=train_1label_TCA_SVM_ROS_shap(xs,ys, xt,yt,i)
        # shap+有precision
        recall[i], f1[i], prc[i], global_shap[i]=train_1label_TCA_SVM_ROS_shap(xs,ys, xt,yt,i)
    '''标准数据集划分end'''	
    # 保存预测后结果
    recall = pd.DataFrame(recall)
    f1 = pd.DataFrame(f1)
    prc = pd.DataFrame(prc)
    global_shap = pd.DataFrame(global_shap)
    performance = pd.concat([f1, recall, prc], axis=1, sort=False)
    # 设置表头
    performance.columns = ['F1','Recall','Precision']
    performance.to_csv('./data/imbalance_experiment/{}TCA+SVM_ROS_performance.csv'.format(curtime), index=False)   
    global_shap.to_csv('./data/wrapper_experiment/{}TCA+SVM_wrapper_shap.csv'.format(curtime), index=False)
    print('TCA Performance saved')

# 上采样无shap
def TCA_SVM_cross_val_ROS_unshap_train(label_num=9):
    # train_data = pd.read_csv('./data/TCA_source_data.csv')
    # test_data = pd.read_csv('./data/TCA_target_data.csv')
    # source_data_lable = pd.read_csv('./data/fill_labels.csv')
    # target_data_lable = source_data_lable.copy(deep=True)
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
    # for i in range(5):
        # 无shap+有precision
        recall[i], f1[i], prc[i]=train_1label_TCA_SVM_ROS_unshap(xs,ys, xt,yt,i)
        # return_values=train_1label_TCA_SVM_ROS_unshap(xs,ys, xt,yt,i)
        # print("return_values:",return_values)
        # # shap+有precision
        # recall[i], f1[i], prc[i], global_shap[i]=train_1label_TCA_SVM_ROS_shap(xs,ys, xt,yt,i)
    '''标准数据集划分end'''	
    # 保存预测后结果
    recall = pd.DataFrame(recall)
    f1 = pd.DataFrame(f1)
    prc = pd.DataFrame(prc)
    # global_shap = pd.DataFrame(global_shap)
    performance = pd.concat([f1, recall, prc], axis=1, sort=False)
    # 设置表头
    performance.columns = ['F1','Recall','Precision']
    performance.to_csv('./data/imbalance_experiment/{}TCA+SVM_ROS_unshap_performance.csv'.format(curtime), index=False)   
    # global_shap.to_csv('./data/wrapper_experiment/{}TCA+SVM_wrapper_shap.csv'.format(curtime), index=False)
    print('TCA Performance saved')

# 测试集一起上采样
def TCA_SVM_cross_val_ROS_testdata_train(label_num=9): 
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
        recall[i], f1[i], prc[i]=train_1label_TCA_SVM_ROS_testdata(xs,ys, xt,yt,i)
    '''标准数据集划分end'''	
    # 保存预测后结果
    recall = pd.DataFrame(recall)
    f1 = pd.DataFrame(f1)
    prc = pd.DataFrame(prc)
    # global_shap = pd.DataFrame(global_shap)
    recall.to_csv('./data/imbalance_experiment/{}TCA+SVM_ROS_testdata_recall.csv'.format(curtime), index=False)
    f1.to_csv('./data/imbalance_experiment/{}TCA+SVM_ROS_testdata_f1.csv'.format(curtime), index=False)
    prc.to_csv('./data/imbalance_experiment/{}TCA+SVM_ROS_testdata_prc.csv'.format(curtime), index=False)
    # global_shap.to_csv('./data/wrapper_experiment/{}TCA+SVM_wrapper_shap.csv'.format(curtime), index=False)
    print('TCA Performance saved')

# 下采样
def TCA_SVM_cross_val_RUS_train(label_num=9): 
    xs,ys, xt,yt=TCA_data_read()
    # 查看形状
    print("shape of X:{}\nshape of Y:{}".format(xs.shape,ys.shape))

    '''标准数据集划分'''
    feature_num=xs.shape[1]
    recall = np.zeros(label_num) 
    f1 = np.zeros(label_num) 
    prc= np.zeros(label_num)
    global_shap=np.empty(shape=(label_num,feature_num))
    for i in range(label_num):
        # recall[i], f1[i], global_shap[i]=train_1label_TCA_SVM_wrapper_shap(xs,ys, xt,yt,i)

        # 无shap+有precision
        recall[i], f1[i], prc[i]=train_1label_TCA_SVM_RUS_unshap(xs,ys, xt,yt,i)
    '''标准数据集划分end'''	
    # 保存预测后结果
    recall = pd.DataFrame(recall)
    f1 = pd.DataFrame(f1)
    prc = pd.DataFrame(prc)
    global_shap = pd.DataFrame(global_shap)
    recall.to_csv('./data/imbalance_experiment/{}TCA+SVM_Undersampling_recall.csv'.format(curtime), index=False)
    f1.to_csv('./data/imbalance_experiment/{}TCA+SVM_Undersampling_f1.csv'.format(curtime), index=False)
    prc.to_csv('./data/imbalance_experiment/{}TCA+SVM_Undersampling_prc.csv'.format(curtime), index=False)
    # global_shap.to_csv('./data/wrapper_experiment/{}TCA+SVM_wrapper_shap.csv'.format(curtime), index=False)
    print('TCA Performance saved')

# 新目标HDA+交叉验证+shap
def HDA_SVM_cross_val_shap_train(label_num=9):
    # train_data = pd.read_csv('./data/TCA_source_data.csv',index_col=0)
    # test_data = pd.read_csv('./data/TCA_target_data.csv',index_col=0)
    train_data = pd.read_csv('./data/HDA_s_data.csv')
    test_data = pd.read_csv('./data/HDA_t_data.csv')
    print("Ss.shape",train_data.shape)
    print("St.shape",test_data.shape)
    # 拼接源域目标域
    data = np.vstack((train_data, test_data))
    label = np.vstack((train_data_lable, test_data_lable))

    recall = np.zeros(label_num) 
    f1 = np.zeros(label_num) 
    global_shap=np.empty(shape=(label_num,50))
    for i in range(label_num):
        # recall[i], f1[i]=train_SVM_cross_val(data,label,i)
        recall[i], f1[i], global_shap[i]=train_1label_PCA_SVM_wrapper_shap(data,label,i)	
    # 保存预测后结果
    recall = pd.DataFrame(recall)
    f1 = pd.DataFrame(f1)
    global_shap = pd.DataFrame(global_shap)
    # recall.to_csv('./data/1.7features_HDA+SVM_cross_val_recall.csv', index=False)
    recall.to_csv('./data/{}HDA+SVM_cross_val_recall.csv'.format(curtime), index=False)
    f1.to_csv('./data/{}HDA+SVM_cross_val_f1.csv'.format(curtime), index=False)
    global_shap.to_csv('./data/{}HDA+SVM_cross_val_shap.csv'.format(curtime), index=False)
    # f1.to_csv('./data/1.7features_HDA+SVM_cross_val_f1.csv', index=False)
    print('HDA Performance saved')

# 新目标HDA+交叉验证+shap
def HDA_SVM_cross_val_ROS_train(label_num=9):
    # train_data = pd.read_csv('./data/TCA_source_data.csv',index_col=0)
    # test_data = pd.read_csv('./data/TCA_target_data.csv',index_col=0)
    train_data = pd.read_csv('./data/HDA_s_data.csv')
    test_data = pd.read_csv('./data/HDA_t_data.csv')
    print("Ss.shape",train_data.shape)
    print("St.shape",test_data.shape)
    # 拼接源域目标域
    data = np.vstack((train_data, test_data))
    label = np.vstack((train_data_lable, test_data_lable))

    recall = np.zeros(label_num) 
    f1 = np.zeros(label_num) 
    prc= np.zeros(label_num)

    # global_shap=np.empty(shape=(label_num,50))
    for i in range(label_num):
        # recall[i], f1[i]=train_SVM_cross_val(data,label,i)
        # recall[i], f1[i], global_shap[i]=train_1label_PCA_SVM_wrapper_shap(data,label,i)	
        recall[i], f1[i], prc[i]=train_1label_HDA_SVM_ROS(data,label,i)	
    # 保存预测后结果
    recall = pd.DataFrame(recall)
    f1 = pd.DataFrame(f1)
    prc = pd.DataFrame(prc)
    performance = pd.concat([f1, recall, prc], axis=1, sort=False)
    # 设置表头
    performance.columns = ['F1','Recall','Precision']

    # global_shap = pd.DataFrame(global_shap)
    # recall.to_csv('./data/{}HDA+SVM_cross_val_recall.csv'.format(curtime), index=False)
    # f1.to_csv('./data/{}HDA+SVM_cross_val_f1.csv'.format(curtime), index=False)
    # global_shap.to_csv('./data/{}HDA+SVM_cross_val_shap.csv'.format(curtime), index=False)
    performance.to_csv('./data/imbalance_experiment/{}HDA+SVM_ROS_performance.csv'.format(curtime), index=False)


    # f1.to_csv('./data/1.7features_HDA+SVM_cross_val_f1.csv', index=False)
    print('HDA Performance saved')

# 新目标PCA+交叉验证+shap
def PCA_SVM_cross_val_unshap_train(label_num=9):
    features_num=451
    data_origin_features = pd.read_csv('./data/data_needed_features.csv')
    # train_data = data_origin_features
    # test_data = pd.read_csv('./data/HDA_t_data.csv')
    # data=train_data.values
    data=data_origin_features.values    
    label=train_data_lable.values
    # label_num=9
    recall = np.zeros(label_num) 
    f1 = np.zeros(label_num) 
    prc= np.zeros(label_num)
    global_shap=np.empty(shape=(label_num,features_num))
    for i in range(label_num):
        # recall[i], f1[i]=train_SVM_cross_val(data,label,i)
        # recall[i], f1[i], global_shap[i]=train_1label_PCA_SVM_wrapper_shap(data,label,i)	
        recall[i], f1[i], prc[i]=train_1label_PCA_SVM_wrapper_shap(data,label,i)	
    # 保存预测后结果
    recall = pd.DataFrame(recall)
    f1 = pd.DataFrame(f1)
    prc = pd.DataFrame(prc)
    global_shap = pd.DataFrame(global_shap)
    recall.to_csv('./data/wrapper_experiment/{}PCA+SVM_cross_val_recall.csv'.format(curtime), index=False)
    f1.to_csv('./data/wrapper_experiment/{}PCA+SVM_cross_val_f1.csv'.format(curtime), index=False)
    prc.to_csv('./data/wrapper_experiment/{}PCA+SVM_wrapper_prc.csv'.format(curtime), index=False)
    # global_shap.to_csv('./data/wrapper_experiment/{}PCA+SVM_wrapper_shap.csv'.format(curtime), index=False)
    print('PCA Performance saved')

# PCA交叉验证+上采样
def PCA_SVM_cross_val_ROS_train(label_num=9):
    features_num=451
    data_origin_features = pd.read_csv('./data/data_needed_features.csv')
    # train_data = data_origin_features
    # test_data = pd.read_csv('./data/HDA_t_data.csv')
    # data=train_data.values
    data=data_origin_features.values    
    label=train_data_lable.values
    # label_num=9
    recall = np.zeros(label_num) 
    f1 = np.zeros(label_num) 
    prc= np.zeros(label_num)
    global_shap=np.empty(shape=(label_num,features_num))
    for i in range(label_num):
        # recall[i], f1[i], prc[i]=train_1label_PCA_SVM_wrapper_shap(data,label,i)	
        recall[i], f1[i], prc[i]=train_1label_PCA_SVM_ROS(data,label,i)	
    # 保存预测后结果
    recall = pd.DataFrame(recall)
    f1 = pd.DataFrame(f1)
    prc = pd.DataFrame(prc)
    # global_shap = pd.DataFrame(global_shap)
    performance = pd.concat([f1, recall, prc], axis=1, sort=False)
    # 设置表头
    performance.columns = ['F1','Recall','Precision']
    # recall.to_csv('./data/imbalance_experiment/{}PCA+SVM_ROS_recall.csv'.format(curtime), index=False)
    # f1.to_csv('./data/imbalance_experiment/{}PCA+SVM_ROS_f1.csv'.format(curtime), index=False)
    # prc.to_csv('./data/imbalance_experiment/{}PCA+SVM_ROS_prc.csv'.format(curtime), index=False)
    performance.to_csv('./data/imbalance_experiment/{}PCA+SVM_ROS_performance.csv'.format(curtime), index=False)
    print('PCA Performance saved')

# LASSO交叉验证+上采样
def LASSO_cross_val_ROS_train(label_num=9):
    features_num=451
    n_components=50
    data_origin_features = pd.read_csv('./data/data_needed_features.csv')
    # train_data = data_origin_features
    # test_data = pd.read_csv('./data/HDA_t_data.csv')
    # data=train_data.values
    data=data_origin_features.values    
    label=train_data_lable.values
    feature_names=pd.read_csv('./data/important_features_name_en.csv')
    # label_num=9
    recall = np.zeros(label_num) 
    f1 = np.zeros(label_num) 
    prc= np.zeros(label_num)
    # global_shap=np.empty(shape=(label_num,features_num))
    top_features=np.empty(shape=(label_num,n_components)).astype(np.str_)
    for i in range(label_num):
        # recall[i], f1[i], prc[i]=train_1label_PCA_SVM_wrapper_shap(data,label,i)	
        # recall[i], f1[i], prc[i]=train_1label_PCA_SVM_ROS(data,label,i)	
        recall[i], f1[i], prc[i], top_features[i]=train_1label_LASSO_ROS(data,label,feature_names,n_components,i)	
    # 保存预测后结果
    recall = pd.DataFrame(recall)
    f1 = pd.DataFrame(f1)
    prc = pd.DataFrame(prc)
    # global_shap = pd.DataFrame(global_shap)
    performance = pd.concat([f1, recall, prc], axis=1, sort=False)
    top_features=pd.DataFrame(top_features)
    # 设置表头
    performance.columns = ['F1','Recall','Precision']
    # recall.to_csv('./data/imbalance_experiment/{}PCA+SVM_ROS_recall.csv'.format(curtime), index=False)
    # f1.to_csv('./data/imbalance_experiment/{}PCA+SVM_ROS_f1.csv'.format(curtime), index=False)
    # prc.to_csv('./data/imbalance_experiment/{}PCA+SVM_ROS_prc.csv'.format(curtime), index=False)
    performance.to_csv('./data/output/LASSO_experiment/{}LASSO_ROS_performance.csv'.format(curtime), index=False)
    top_features.to_csv('./data/output/LASSO_experiment/{}LASSO_ROS_top_features.csv'.format(curtime), index=False)
    print('LASSO Performance saved')


# LASSO交叉验证+SVM上采样
def LASSO_SVM_cross_val_ROS_train(label_num=9):
    features_num=451
    n_components=50
    data_origin_features = pd.read_csv('./data/data_needed_features.csv')
    # train_data = data_origin_features
    # test_data = pd.read_csv('./data/HDA_t_data.csv')
    # data=train_data.values
    data=data_origin_features.values    
    label=train_data_lable.values
    feature_names=pd.read_csv('./data/important_features_name_en.csv')
    # label_num=9
    recall = np.zeros(label_num) 
    f1 = np.zeros(label_num) 
    prc= np.zeros(label_num)
    # global_shap=np.empty(shape=(label_num,features_num))
    top_features=np.empty(shape=(label_num,n_components)).astype(np.str_)
    for i in range(label_num):
        # recall[i], f1[i], prc[i]=train_1label_PCA_SVM_wrapper_shap(data,label,i)	
        # recall[i], f1[i], prc[i]=train_1label_PCA_SVM_ROS(data,label,i)	
        recall[i], f1[i], prc[i], top_features[i]=train_1label_LASSO_SVM_ROS(data,label,feature_names,n_components,i)	
    # 保存预测后结果
    recall = pd.DataFrame(recall)
    f1 = pd.DataFrame(f1)
    prc = pd.DataFrame(prc)
    # global_shap = pd.DataFrame(global_shap)
    performance = pd.concat([f1, recall, prc], axis=1, sort=False)
    top_features=pd.DataFrame(top_features)
    # 设置表头
    performance.columns = ['F1','Recall','Precision']
    # recall.to_csv('./data/imbalance_experiment/{}PCA+SVM_ROS_recall.csv'.format(curtime), index=False)
    # f1.to_csv('./data/imbalance_experiment/{}PCA+SVM_ROS_f1.csv'.format(curtime), index=False)
    # prc.to_csv('./data/imbalance_experiment/{}PCA+SVM_ROS_prc.csv'.format(curtime), index=False)
    performance.to_csv('./data/output/LASSO_experiment/{}LASSO_SVM_ROS_performance.csv'.format(curtime), index=False)
    # top_features.to_csv('./data/output/LASSO_experiment/{}LASSO_ROS_top_features.csv'.format(curtime), index=False)
    print('LASSO Performance saved')


# 新目标EF+TCA+交叉验证
def EF_TCA_SVM_cross_val_shap_train(label_num=9):
    # train_data = pd.read_csv('./data/TCA_source_data.csv')
    # test_data = pd.read_csv('./data/TCA_target_data.csv')
    # # 拼接源域目标域
    # data = np.vstack((train_data, test_data))
    label = np.vstack((train_data_lable, test_data_lable))
    # label_num=9
    recall = np.zeros(label_num) 
    f1 = np.zeros(label_num) 
    global_shap=np.empty(shape=(label_num,50))
    for i in range(label_num):
        data = pd.read_csv('./data/{}_fity{}.csv'.format('TCA_after_EF_data',i))
        data=data.values
        # recall[i], f1[i]=train_SVM_cross_val(data,label,i)
        recall[i], f1[i], global_shap[i]=train_1label_PCA_SVM_wrapper_shap(data,label,i)
        return_values=train_1label_PCA_SVM_wrapper_shap(data,label,i)
        print("return_values:",return_values)
    # 保存预测后结果
    recall = pd.DataFrame(recall)
    f1 = pd.DataFrame(f1)
    global_shap = pd.DataFrame(global_shap)
    recall.to_csv('./data/{}EF+TCA+SVM_cross_val_recall.csv'.format(curtime), index=False)
    f1.to_csv('./data/{}EF+TCA+SVM_cross_val_f1.csv'.format(curtime), index=False)
    global_shap.to_csv('./data/{}EF+TCA+SVM_cross_val_shap.csv'.format(curtime), index=False)
    print('EF+TCA Performance saved')


if __name__ == '__main__':
    # TCA_SVM_cross_val_RUS_train(9)#下采样
    # TCA_SVM_cross_val_ROS_testdata_train(9)#测试集上采样
    # PCA_SVM_cross_val_unshap_train(9)
    # EF_TCA_SVM_cross_val_shap_train(9)
    # HDA_SVM_cross_val_shap_train(9)
    # TCA_SVM_cross_val_ROS_shap_train(9)#训练集上采样+shap
    # TCA_SVM_cross_val_ROS_unshap_train(9)#训练集上采样无shap
    # PCA_SVM_cross_val_ROS_train(9)#PCA上采样
    LASSO_cross_val_ROS_train(9)#LASSO上采样
    # LASSO_SVM_cross_val_ROS_train(9)#LASSO上采样
    # HDA_SVM_cross_val_ROS_train(9)

    

    

