# import time
# curtime = time.strftime('%Y-%m-%d',time.localtime(time.time()))
# print('./data/{}features_EF+TCA+SVM_cross_val_recall.csv'.format(curtime))
# # recall.to_csv('./data/{}features_EF+TCA+SVM_cross_val_recall.csv'.format(curtime), index=False)
import numpy as np
from sklearn import decomposition
'''sklearn PCA以及逆转换'''
# 建立简单矩阵
# X = np.array([[-1, -1, -1], [-2, -1, -1], [-3, -2, 3], [1, 1, -3], [2, 1, 4], [3, 2, -5]])
X = np.array([[1, -1, -1], [2, -1, -1], [3, -2, 3], [4, 1, -3], [5, 1, 4], [6, 2, -5]])
# sum=np.array([1,2,3,4,5,6])
global_shap = np.array([[0.8,0.2]])
# # 标准化
# scaler = preprocessing.StandardScaler()
# X_data = scaler.fit_transform(X_data)
# 将含有2个特征的数据经过PCA压缩为1个特征
pca = decomposition.PCA(n_components=1)
pca.fit(X)
X_pca = pca.transform(X)
# print("X_pca:\n",X_pca)
X_origin=pca.inverse_transform(X_pca)
# print("X_origin:\n",X_origin)
'''手刻PCA及逆转换'''
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
    return new_data,feature,mean
# X_pca = pca(X, 1)#可行
X_pca,X_components,X_mean = pca(X, 2)
# print("X_pca:\n{}\nX_components:\n{}".format(X_pca,X_components))
X_origin=np.dot(X_pca, X_components)+X_mean
use_ratio=np.maximum(X_components,-X_components)
shap_origin=np.dot(global_shap, use_ratio)
# print("X_origin:\n{}\nshap_origin:\n{}".format(X_origin, shap_origin))
sum=X.sum(axis=1)
# sum=sum.reshape((len(sum),1))
# ratio=X / sum
# print("sum:\n{}\n".format(sum))
# print("sum:\n{}\nX / sum=\n{}".format(sum,X / sum))



x = np.array([[1000,  10,   0.5],
              [ 765,   5,  0.35],
              [ 800,   7,  0.09],
              [ 800,   7,  0.09]])
# print(x.sum(axis=0))
sum=x.sum(axis=1)
sum=sum.reshape((len(sum),1))
x_normed = x / sum

print(x_normed)