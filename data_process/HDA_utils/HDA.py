# coding=utf-8

import pandas as pd
import numpy as np
from sklearn import preprocessing
from scipy.spatial.distance import pdist
import importlib
# learn_coding = importlib.import_module('data_process.HDA_utils.learn_coding')
# learn_dictionary = importlib.import_module('data_process.HDA_utils.learn_dictionary')

import learn_coding
import learn_dictionary
import scipy.io as sio

mu = 10000          # MMD regularization
alpha = 1            # graph regularization
lamda = 0.01         # sparsity regularization
# k = 51             #number of basis vectors
k = 50             #number of basis vectors
gamma = 0.2
result = []

#加载源域目标域数据
print('------读取源域数据、目标域数据---------')
# Xs = pd.read_csv('../data/source_data.csv',index_col=0)
# Xt = pd.read_csv('../data/target_data.csv')
# Ys = pd.read_csv('../data/source_data_lable.csv')
# Yt = pd.read_csv('../data/target_data_lable.csv')
Xs = pd.read_csv('./data/source_data.csv')
Xt = pd.read_csv('./data/target_data.csv')
Ys = pd.read_csv('./data/fill_labels.csv')
Yt =  Ys.copy(deep=True)



print(Xs.shape)
print(Xt.shape)
print(Ys.shape)
print(Yt.shape)

def pca(X, k): # k is the components you want
    # mean of each feature
    n_samples, n_features = X.shape
    mean = np.array([np.mean(X[:, i]) for i in range(n_features)])
    # normalization
    norm_X = X - mean
    # scatter matrix
    scatter_matrix = np.dot(np.transpose(norm_X), norm_X)
    # Calculate the eigenvectors and eigenvalues
    eig_val, eig_vec = np.linalg.eig(scatter_matrix)
    eig_pairs = [(np.abs(eig_val[i]), eig_vec[:, i]) for i in range(n_features)]
    # sort eig_vec based on eig_val from highest to lowest
    eig_pairs = [((np.abs(eig_val[i])), eig_vec[:, i]) for i in range(len(eig_vec))]
    eig_pairs.sort(key=lambda x: x[0], reverse=True)
    # select the top k eig_vec
    feature = np.array([ele[1] for ele in eig_pairs[:k]])
    # get new data
    new_data = np.dot(norm_X, np.transpose(feature))
    return new_data

def cosine_distance(vec1, vec2):  # 余弦相似度
    Vec = np.vstack([vec1, vec2])
    dist2 = 1 - pdist(Vec, 'cosine')

    return dist2



# Normalization of original data
scaler = preprocessing.StandardScaler()
Xs_scaled = scaler.fit_transform(Xs)
Xt_scaled = scaler.fit_transform(Xt)

print('--------Perform PCA on original data-------')
newXs = pca(Xs.values, 60)
newXt = pca(Xt.values,60)

print(newXs.shape)
print(newXt.shape)

X = np.hstack((newXs.transpose(), newXt.transpose()))

print('X.shape为：')
print(X.shape)

print('----------初始化B-----------')
B = np.random.randn(X.shape[0],k)

print('B.shape为：')
print(B.shape)
print(B)

print('----------计算W矩阵-----------')
def kn(X, k):
    value=[]
    X = np.array(X)

    for j in range(np.shape(X)[0]):
        x_j = X[j]
        res = []
        for m in range(np.shape(X)[0]):
            x_m = X[m]
            x_m = np.array(x_m)
            x_j = np.array(x_j)
            distance = np.mean(np.sqrt((x_j - x_m) ** 2))
            res.append(distance)

        b = np.argsort(res)
        b = b[:k]
        value.append(b)
    return value

if __name__=='__main__':
    X = list(np.array(X).T)
    X = np.array(X)
    w_ij = np.zeros((np.shape(X)[0], np.shape(X)[0]))
    value = kn(X, 41)
    for i in range(np.shape(X)[0]):
        x_i = X[i]
        for j in range(len(value)):
            if i in value[j]:
                w_ij[i][j] = cosine_distance(X[j], x_i)
            else:
                w_ij[i][j] = 0

    print("w：", w_ij)
    print('W.shape为：')
    print(w_ij.shape)

print('----------计算L矩阵-----------')
a = w_ij.sum(axis=1)        #是计算矩阵的每一行元素相加之和。
D = np.diag(a)
print(D)
print(D.shape)

L = D-w_ij
print('L.shape为：')
print(L.shape)

print('----------计算M矩阵-----------')

ns = newXs.shape[0]
print(ns)
nt = newXt.shape[0]
print(nt)
e1 = np.hstack([(1/ns)*np.ones(ns),(1/nt)*np.ones(nt)])
e1 = pd.DataFrame(e1)
print(e1.shape)
print(e1.T.shape)
M = np.dot(e1, e1.T)
print('M.shape为：')
print(M.shape)

print('------------开始迭代求B、S---------------')

Sinit = np.random.randn(k,X.shape[1])
Sinit1= Sinit.tolist()

for i in range(10):
    S = learn_coding.learn_coding(B.T, X, alpha)
    Ss = S[:,:425]
    St = S[:, 425:]
    B = learn_dictionary.lagrange_dual_learn( X.T, S, 1,c_const = 0.001)

print('———————————————————将处理的结果S进行保存————————————————————————')

# np.savetxt("HDA_s_data.csv", Ss, delimiter=',')
# np.savetxt("HDA_t_data.csv", St, delimiter=',')
Ss = pd.DataFrame(Ss.T)
St = pd.DataFrame(St.T)
Ss.to_csv('./data/HDA_s_data.csv', index=False)
St.to_csv('./data/HDA_t_data.csv', index=False)

print("Ss.shape",Ss.shape)
print("St.shape",St.shape)

print("Ss",Ss)
print("St",St)

