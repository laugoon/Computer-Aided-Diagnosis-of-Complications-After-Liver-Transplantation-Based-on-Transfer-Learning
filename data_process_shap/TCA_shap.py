# coding: utf-8

import pandas as pd
import numpy as np
import scipy.linalg
import sklearn.metrics

#add
from sklearn.model_selection import train_test_split


# source_data = pd.read_csv('./data/source_data.csv')
# target_data = pd.read_csv('./data/target_data.csv')
# source_data_lable = pd.read_csv('./data/fill_labels.csv')
# # 防止浅拷贝问题
# target_data_lable = source_data_lable.copy(deep=True)
# # target_data_lable = pd.read_csv('./data/target_data_lable.csv',index_col=0)


# print(source_data.shape )
# print(target_data.shape )
# print(source_data_lable.shape)
# print(target_data_lable.shape)

def kernel(ker, X1, X2, gamma):
    K = None
    if not ker or ker == 'primal':
        K = X1
    elif ker == 'linear':
        if X2 is not None:
            K = sklearn.metrics.pairwise.linear_kernel(np.asarray(X1).T, np.asarray(X2).T)
        else:
            K = sklearn.metrics.pairwise.linear_kernel(np.asarray(X1).T)
    elif ker == 'rbf':
        if X2 is not None:
            K = sklearn.metrics.pairwise.rbf_kernel(np.asarray(X1).T, np.asarray(X2).T, gamma)
        else:
            K = sklearn.metrics.pairwise.rbf_kernel(np.asarray(X1).T, None, gamma)
    return K


class TCA:
    def __init__(self, kernel_type='linear', dim = 30, lamb = 1, gamma=1):
        '''
        Init func
        :param kernel_type: kernel, values: 'primal' | 'linear' | 'rbf'
        :param dim: dimension after transfer
        :param lamb: lambda value in equation
        :param gamma: kernel bandwidth for rbf kernel
        '''
        self.kernel_type = kernel_type
        self.dim = dim
        self.lamb = lamb
        self.gamma = gamma
        self.A = None

    def fit(self, Xs, Xt):
        '''
        Transform Xs and Xt
        :param Xs: ns * n_feature, source feature
        :param Xt: nt * n_feature, target feature
        :return: Xs_new and Xt_new after TCA
        '''
        # print(Xs.shape)
        # print(Xt.shape)
        X = np.hstack((Xs.T, Xt.T))
        #范数
        X /= np.linalg.norm(X, axis=0)
        m, n = X.shape
        #print(m,n)
        ns, nt = len(Xs), len(Xt)
        #print(ns,nt)
        e = np.vstack((1 / ns * np.ones((ns, 1)), -1 / nt * np.ones((nt, 1))))
        L = e * e.T
        L = L / np.linalg.norm(L, 'fro')
        H = np.eye(n) - 1 / n * np.ones((n, n))
        K = kernel(self.kernel_type, X, None, gamma=self.gamma)
        n_eye = m if self.kernel_type == 'primal' else n
        # a, b = np.linalg.multi_dot([K, L, K.T]) + self.lamb * np.eye(n_eye), np.linalg.multi_dot([K, H, K.T])
        lambI=self.lamb* np.eye(n_eye)
        KLK=np.linalg.multi_dot([K, L, K.T])
        a = KLK + lambI
        b = np.linalg.multi_dot([K, H, K.T])
        w, V = scipy.linalg.eig(a, b)#w特征值，V特征向量

        ind = np.argsort(w)
        A = V[:, ind[:self.dim]]#变换矩阵
        self.A = A#保存转换矩阵

        Z = np.dot(A.T, K)#隐空间坐标
        Z /= np.linalg.norm(Z, axis=0)
        #print(Z.shape)
        Xs_new, Xt_new = Z[:, :ns].T, Z[:, ns:].T
        print(Xs_new.shape, Xt_new.shape)
        return Xs_new, Xt_new
        
    # def fit_new(self, Xs, Xt, Xt2):
    #     '''
    #     Map Xt2 to the latent space created from Xt and Xs
    #     :param Xs : ns * n_feature, source feature
    #     :param Xt : nt * n_feature, target feature
    #     :param Xt2: n_s, n_feature, target feature to be mapped
    #     :return: Xt2_new, mapped Xt2 with projection created by Xs and Xt
    #     '''
    #     # Computing projection matrix A from Xs an Xt
    #     X = np.hstack((Xs.T, Xt.T))
    #     X /= np.linalg.norm(X, axis=0)

    #     #计算转换矩阵A的过程
    #     # m, n = X.shape
    #     # ns, nt = len(Xs), len(Xt)
    #     # e = np.vstack((1 / ns * np.ones((ns, 1)), -1 / nt * np.ones((nt, 1))))
    #     # M = e * e.T
    #     # M = M / np.linalg.norm(M, 'fro')
    #     # H = np.eye(n) - 1 / n * np.ones((n, n))
    #     # K = kernel(self.kernel_type, X, None, gamma=self.gamma)    
    #     # n_eye = m if self.kernel_type == 'primal' else n
    #     # a, b = np.linalg.multi_dot([K, M, K.T]) + self.lamb * np.eye(n_eye), np.linalg.multi_dot([K, H, K.T])
    #     # w, V = scipy.linalg.eig(a, b)
    #     # ind = np.argsort(w)#将数组的值从小到大排序后,并按照其相对应的索引值输出 一维数组
    #     # A = V[:, ind[:self.dim]]
    #     # 计算转换矩阵A的过程end
    #     A = self.A#读取转换矩阵

        
    #     # Compute kernel with Xt2 as target and X as source
    #     Xt2 = Xt2.T
    #     Xt2 /= np.linalg.norm(Xt2, axis=0)#个人增加
    #     K = kernel(self.kernel_type, X1 = Xt2, X2 = X, gamma=self.gamma)#保持了维度一致
        
    #     # New target features
    #     Xt2_new = K @ A
        
    #     return Xt2_new
    '''修改规范化'''
    def fit_new(self, Xs, Xt, Xt2):
        '''
        Map Xt2 to the latent space created from Xt and Xs
        :param Xs : ns * n_feature, source feature
        :param Xt : nt * n_feature, target feature
        :param Xt2: n_s, n_feature, target feature to be mapped
        :return: Xt2_new, mapped Xt2 with projection created by Xs and Xt
        '''
        # Computing projection matrix A from Xs an Xt
        X = np.hstack((Xs.T, Xt.T))
        X /= np.linalg.norm(X, axis=0)
        # 训练集样本个数
        n_train = len(Xs)+len(Xt)

        # 原代码
        # m, n = X.shape
        # ns, nt = len(Xs), len(Xt)
        # e = np.vstack((1 / ns * np.ones((ns, 1)), -1 / nt * np.ones((nt, 1))))
        # M = e * e.T
        # M = M / np.linalg.norm(M, 'fro')
        # H = np.eye(n) - 1 / n * np.ones((n, n))
        # K = kernel(self.kernel_type, X, None, gamma=self.gamma)
        # n_eye = m if self.kernel_type == 'primal' else n
        # a, b = np.linalg.multi_dot([K, M, K.T]) + self.lamb * np.eye(n_eye), np.linalg.multi_dot([K, H, K.T])
        # w, V = scipy.linalg.eig(a, b)
        # ind = np.argsort(w)#将数组的值从小到大排序后,并按照其相对应的索引值输出 一维数组
        # A = V[:, ind[:self.dim]]
        # 原代码end

        '''修改规范化'''
        # # 转换矩阵A不再重新计算
        # A = self.A#读取转换矩阵
        # # 测试集样本个数
        # # n_test = len(Xt2)
        # # Compute kernel with Xt2 as target and X as source
        # Xt2 = Xt2.T
        # # print("规范化前Xt2范数:\n",np.linalg.norm(Xt2, axis=0))
        # # 新样本(测试集)个数

        # X_Xt2 = np.hstack((Xs.T, Xt.T,Xt2))
        # X_Xt2/=np.linalg.norm(X_Xt2, axis=0)
        # Xt2 = X_Xt2[:, n_train:]
        # # print("规范化后Xt2范数:\n",np.linalg.norm(Xt2, axis=0))
        # print("\nX1形状:",Xt2.shape,"\nX2形状:",X.shape)
        # K = kernel(self.kernel_type, X1 = Xt2, X2 = X, gamma=self.gamma)
        
        # # New target features
        # Xt2_new = K @ A
        '''修改规范化end'''

        '''修改核矩阵中样本个数'''
        # 转换矩阵A不再重新计算
        A = self.A#读取转换矩阵
        # 测试集样本个数
        # n_test = len(Xt2)
        # Compute kernel with Xt2 as target and X as source
        Xt2 = Xt2.T
        # print("规范化前Xt2范数:\n",np.linalg.norm(Xt2, axis=0))
        

        # X已经经过规范化, 不能作为原始数据用
        X_Xt2 = np.hstack((Xs.T, Xt.T,Xt2))
        X_Xt2/=np.linalg.norm(X_Xt2, axis=0)
        # Xt2 = X_Xt2[:, n_train:]
        # # print("规范化后Xt2范数:\n",np.linalg.norm(Xt2, axis=0))
        # print("\nX1形状:",Xt2.shape,"\nX2形状:",X.shape)
        K = kernel(self.kernel_type, X1 = X_Xt2, X2 = X, gamma=self.gamma)
        
        # New target features
        # Xt2_new = K @ A
        X_Xt2_new = K @ A
        X_Xt2_new=X_Xt2_new.T
        X_Xt2_new/= np.linalg.norm(X_Xt2_new, axis=0)
        Xt2_new = X_Xt2_new[:,n_train:].T


        # Z = np.dot(A.T, K)
        # Z /= np.linalg.norm(Z, axis=0)
        
        # Xs_new, Xt_new = Z[:, :ns].T, Z[:, ns:].T
        '''修改核矩阵中样本个数end'''
        return Xt2_new

'''类方法end'''

def TCA_data_read():
    source_data = pd.read_csv('./data/source_data.csv')
    target_data = pd.read_csv('./data/target_data.csv')
    source_data_lable = pd.read_csv('./data/fill_labels.csv')
    # 防止浅拷贝问题
    target_data_lable = source_data_lable.copy(deep=True)
    # print("source_data Colum0:\n",source_data.values[:,0])#确认没有读到index列

    # # 分割测试集
    # xt_train, xt_test, yt_train, yt_test = train_test_split(target_data, target_data_lable, test_size=0.4, random_state=0)
    # 数据类型转换dataframe->ndarray
    return source_data.values,source_data_lable.values, target_data.values,target_data_lable.values

def TCA_after_fillNA():
    source_data,ys, target_data,yt=TCA_data_read()
    # # TCA_data_read
    # source_data = pd.read_csv('./data/source_data.csv')
    # target_data = pd.read_csv('./data/target_data.csv')
    # source_data_lable = pd.read_csv('./data/fill_labels.csv')
    # # 防止浅拷贝问题
    # target_data_lable = source_data_lable.copy(deep=True)
    # # TCA_data_read end 
    
    # 分割测试集
    xt_train, xt_test, yt_train, yt_test = train_test_split(target_data, yt, test_size=0.4, random_state=0)

    print(source_data.shape )
    print(target_data.shape )
    print(yt_train.shape)
    print(yt_test.shape)

    # tca = TCA(kernel_type='rbf', dim=50, lamb=0.9, gamma=0.5)
    # Xs_new, Xt_new = tca.fit(source_data, target_data)
    tca = TCA(kernel_type='rbf', dim=50, lamb=0.9, gamma=0.5)
    Xs_new, Xt_train_new = tca.fit(source_data, xt_train)
    Xt_test_new = tca.fit_new(source_data, xt_train,xt_test)

    Xs_new = pd.DataFrame(Xs_new)
    Xt_train_new = pd.DataFrame(Xt_train_new)
    #print(Xs_new)
    Xs_new.to_csv('./data/TCA_source_data_train.csv', index=False)
    Xt_train_new.to_csv('./data/TCA_target_data_train.csv', index=False)
    print("TCA_after_fillNA data saved")

def TCA_after_EF(label_num=9):
    # source_data = pd.read_csv('./data/source_data.csv')
    # target_data = pd.read_csv('./data/target_data.csv')
    source_data_lable = pd.read_csv('./data/fill_labels.csv')
    # 防止浅拷贝问题
    target_data_lable = source_data_lable.copy(deep=True)
    print(source_data_lable.shape)
    print(target_data_lable.shape)

    for i in range(label_num):
        source_data = pd.read_csv('./data/{}_fity{}.csv'.format('EF_source_data',i))
        target_data = pd.read_csv('./data/{}_fity{}.csv'.format('EF_target_data',i))
        print("source_data.shape:",source_data.shape )
        print("target_data.shape:",target_data.shape )
        tca = TCA(kernel_type='rbf', dim=50, lamb=0.9, gamma=0.5)
        Xs_new, Xt_new = tca.fit(source_data, target_data)
        X_new = np.vstack((Xs_new, Xt_new))
        X_new = pd.DataFrame(X_new)
        # Xs_new = pd.DataFrame(Xs_new)
        # Xt_new = pd.DataFrame(Xt_new)
        X_new.to_csv('./data/{}_fity{}.csv'.format('TCA_after_EF_data',i), index=False)
        # Xs_new.to_csv('./data/{}_fity{}.csv'.format('TCA_after_EF_source_data',i), index=False)
        # Xt_new.to_csv('./data/{}_fity{}.csv'.format('TCA_after_EF_target_data',i), index=False)



    # source_data = pd.read_csv('./data/source_data.csv')
    # target_data = pd.read_csv('./data/target_data.csv')
    # tca = TCA(kernel_type='rbf', dim=50, lamb=0.9, gamma=0.5)
    # Xs_new, Xt_new = tca.fit(source_data, target_data)
    # Xs_new = pd.DataFrame(Xs_new)
    # Xt_new = pd.DataFrame(Xt_new)
    # #print(Xs_new)
    # Xs_new.to_csv('./data/TCA_source_data.csv', index=False)
    # Xt_new.to_csv('./data/TCA_target_data.csv', index=False)

if __name__ == '__main__':
    # source_data,source_data_lable, xt_train,yt_train, xt_test,yt_test=TCA_train_test_split()
    TCA_after_fillNA()
    TCA_after_EF(9)
    