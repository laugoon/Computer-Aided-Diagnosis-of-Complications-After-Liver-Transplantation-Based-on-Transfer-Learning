import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
# import data_split as spl
from sklearn import metrics
from sklearn.metrics import f1_score
# data = pd.read_csv('./data/source_data.csv')
data = pd.read_csv('./data/data_needed_features.csv')

# source_data = pd.read_csv('./data/source_data.csv')
# target_data = pd.read_csv('./data/target_data.csv')
lable = pd.read_csv('./data/fill_labels.csv')
# # 忽略索引列
# data = pd.read_csv('data\do_fill_na.csv',index_col=0)
# train_data = pd.read_csv('./data/TCA_source_data.csv',index_col=0)


###pca降维
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
    # return new_data
    return new_data,feature
if __name__ == '__main__':
    # X,Y,X_test,y_test = spl.data_split(data)
    # important_column = ['身高', '体重', 'BMI', 'O型', 'A型' ,'B型', 'AB型' ,'RH(阳1阴0)', '乙肝携带' ,
    important_column = ['身高', '体重', 'BMI', 'O型', 'A型' ,'B型', 'AB型' , '乙肝携带' ,
    '术式(经典1背驮0)', '手术时间min' ,'无肝期时间min' ,'热缺血时间min' ,'冷缺血时间min', '红细胞', '血浆', '自体血',
    #  '术式(经典1背驮2)', '手术时间min' ,'无肝期时间min' ,'热缺血时间min' ,'冷缺血时间min', '红细胞', '血浆', '自体血',
    '4%白蛋白' ,'2%白蛋白', '纯白蛋白g', 'NS' ,'LR', '万汶' ,'佳乐施', '总入量', '出血量', '胸水' ,'腹水', '总尿量',
    'I期尿量', 'II期尿量' ,'III期尿量', '速尿mg', '甘露醇ml', '碳酸氢钠ml', '纤维蛋白原g' ,'凝血酶原复合物U',
    'VII因子' ,'氨甲环酸g/h', '氨甲环酸入壶g' ,'去甲肾上腺素维持' ,'去甲肾上腺素出室', '肾上腺素维持', '肾上腺素出室',
    '多巴胺维持mg/h' ,'多巴胺出室' ,'开放时阿托品' ,'开放时最低心率' ,'开放时最低SBP' ,'开放时最低DBP', '开放时最低MBP',
    '再灌注后综合征', '切脾', '肝肾联合移植' ,'特利加压素ml/h','Hb-pre_x' ,'HCT-pre', 'MCV-pre',
    'MCH-pre' ,'MCHC-pre' ,'RDW-CVO-pre', 'PLT-pre', 'MPV-pre' ,'PDW-pre',
    'LCR-pre', 'Hb-post', 'HCT-post', 'MCV-post' ,'MCH-post', 'MCHC-post',
    'RDW-CVO-post', 'PLT-post' ,'MPV-post' ,'PDW-post', 'LCR-post',
    'Hb-1' ,'HCT-1','MCV-1' ,'MCH-1', 'MCHC-1', 'RDW-CVO-1' ,'PLT-1' ,'MPV-1' ,'PDW-1', 'LCR-1',
    'Hb-3', 'HCT-3', 'MCV-3' ,'MCH-3', 'MCHC-3', 'RDW-CVO-3','PLT-3' ,'MPV-3', 'PDW-3', 'LCR-3',
    'Hb-5', 'HCT-5' ,'MCV-5','MCH-5' ,'MCHC-5', 'RDW-CVO-5', 'PLT-5', 'MPV-5', 'PDW-5', 'LCR-5',
    'Hb-7' ,'HCT-7' ,'MCV-7' ,'MCH-7' ,'MCHC-7' ,'RDW-CVO-7', 'PLT-7','MPV-7' ,'PDW-7', 'LCR-7',
    'AST-pre', 'ALT-pre','TBIL-pre' ,'ALB-pre', 'BUN-pre', 'Cr-pre' ,'Glu-pre_x' ,'K-pre_x' ,'Na-pre_x','Ca-pre_x',
    'AST-post', 'ALT-post' ,'TBIL-post', 'ALB-post' ,'BUN-post','Cr-post', 'K-post', 'Na-post' ,'Ca-post',
    'AST-1', 'ALT-1', 'TBIL-1', 'ALB-1','BUN-1' ,'Cr-1' ,'K-1', 'Na-1' ,'Ca-1',
    'AST-3' ,'ALT-3' ,'TBIL-3', 'ALB-3','BUN-3' ,'Cr-3' ,'K-3', 'Na-3' ,'Ca-3',
    'AST-5', 'ALT-5', 'TBIL-5', 'ALB-5','BUN-5', 'Cr-5' ,'K-5', 'Na-5' ,'Ca-5',
    'AST-7', 'ALT-7', 'TBIL-7' ,'ALB-7','BUN-7' ,'Cr-7' ,'K-7', 'Na-7' ,'Ca-7',
    'PH-pre','PCO2-pre','PO2-pre','Na-pre_y','K-pre_y','Ca-pre_y','Glu-pre_y','Lac-pre','Hct-pre','BE(B)-pre', 'Hb-pre_y',
    'PH-0','PCO2-0','PO2-0','Na-0','K-0','Ca-0','Glu-0','Lac-0','Hct-0','BE(B)-0','Hb-0',
    'PH-60' ,'PCO2-60', 'PO2-60','Na-60','K-60','Ca-60','Glu-60','Lac-60','Hct-60','BE(B)-60','Hb-60',
    'PH-end','PCO2-end','PO2-end','Na-end','K-end','Ca-end','Glu-end','Lac-end','Hct-end','BE(B)-end','Hb-end',
    'PH-1d' ,'PCO2-1d','PO2-1d', 'Na-1d' ,'K-1d' ,'Ca-1d', 'Glu-1d', 'Lac-1d', 'Hct-1d', 'BE(B)-1d','Hb-1d' ,
    'PA-pre','PT-pre','PR-pre','APTT-pre','FBG-pre','INR-pre' ,
    'PA-post','PT-post','PR-post','APTT-post','FBG-post','INR-post','D-Dimer-post',
    'PA-1','PT-1','PR-1','APTT-1','FBG-1','INR-1','D-Dimer-1',
    'PA-3','PT-3', 'PR-3', 'APTT-3' ,'FBG-3' ,'INR-3', 'D-Dimer-3',
    'PA-5', 'PT-5','PR-5' ,'APTT-5' ,'FBG-5' ,'INR-5' ,'D-Dimer-5',
    'PA-7', 'PT-7' ,'PR-7' ,'APTT-7' ,'FBG-7' ,'INR-7',

    '红细胞POD1','红细胞POD3','红细胞POD5','红细胞POD7','红细胞POD9','红细胞POD11','红细胞POD13','红细胞POD14+','术后红细胞总量',
    '血浆POD1','血浆POD3','血浆POD5','血浆POD7','血浆POD9','血浆POD11','血浆POD13','血浆POD14+', '术后血浆总量',
    '血小板POD1','血小板POD3','血小板POD5','血小板POD7','血小板POD9','血小板POD11','血小板POD13','血小板POD14+',
    '术后带管时间h', 'ICU驻留时间d' ,'术后住院时间d' ]
    # X_data = X[important_column]
    # 在分割数据集的时候已经划分过特征了
    X_data = data
    # X_test_data = X_test[important_column]
    # print(X_data.shape)
    # print(X_test_data.shape)

    ##归一化

    scaler = preprocessing.StandardScaler()
    X_data = scaler.fit_transform(X_data)
    # X_test_data = scaler.fit_transform(X_test_data)

    # X_data_pca = pca(X_data, 50).real#.real取得复数的实部
    X_data_pca,X_components = pca(X_data, 50)

    # X_test_data_pca = pca(X_test_data, 50)
    # X_test_data_pca = X_test_data_pca.real

    X_data_pca = pd.DataFrame(X_data_pca)
    X_components = pd.DataFrame(X_components)
    X_data_pca.to_csv('./data/PCA_data.csv', index=False)
    X_components.to_csv('./data/PCA+SVM_components.csv', index=False)
    print('PCA features saved')
# print('------------预测术后并发症I--------------')
# y_data1 = Y['术后并发症I']
# #y_test_data1 = y_test['术后并发症I']
# svm1 = SVC(kernel='rbf', gamma=100.0, C=100.0,  random_state=0)
# #svm1.fit(X_data_pca,y_data1)
# #y_pred1 = svm1.predict(X_test_data_pca)
# #print('准确率: %.2f' % metrics.precision_score(y_test_data1,y_pred1))
# print('X_data_pca:',X_data_pca)
# print('精确率：%.2f' % np.mean(cross_val_score(svm1,X_data_pca,y_data1,cv=3,scoring='recall')))
# print('------------预测术后并发症II--------------')
# y_data2 = Y['术后并发症II']
# #y_test_data2 = y_test['术后并发症II']
# svm2 = SVC(kernel='rbf', C=100.0,gamma=100.0, random_state=0)
# # svm2.fit(X_data_pca,y_data2)
# # y_pred2 = svm2.predict(X_test_data_pca)
# #print(y_pred2)
# #print(y_test_data2)
# #print('准确率: %.2f' % metrics.precision_score(y_test_data2,y_pred2))
# print('精确率：%.2f' % np.mean(cross_val_score(svm2,X_data_pca,y_data2,cv=3,scoring='recall')))





# print('------------预测术后并发症IIIa--------------')
# y_data3 = Y['术后并发症IIIa']
# #y_test_data3 = y_test['术后并发症IIIa']
# svm3 = SVC(kernel='rbf', C=100.0,gamma=100.0, random_state=0)
# # svm3.fit(X_data_pca,y_data3)
# # y_pred3 = svm3.predict(X_test_data_pca)
# # print(y_pred3)
# # print(y_test_data3)
# #print('准确率: %.2f' % np.mean(cross_val_score(svm3,X,y_data3,cv=10,scoring='precision')))
# print('精确率：%.2f' % np.mean(cross_val_score(svm3,X_data_pca,y_data3,cv=3,scoring='recall')))

# print('------------预测术后并发症IIIb--------------')
# y_data4 = Y['术后并发症IIIb']
# #y_test_data4 = y_test['术后并发症IIIb']
# svm4 = SVC(kernel='rbf', C=100.0,gamma=100.0, random_state=0)
# #svm4.fit(X_data_pca, y_data4)
# #y_pred4 = svm4.predict(X_test_data_pca)
# print('准确率: %.2f' % np.mean(cross_val_score(svm4,X_data_pca,y_data4,cv=3,scoring='recall')))
# # print(y_pred4)
# # print(y_test_data4)

# print('------------预测术后并发症IV--------------')
# y_data5 = Y['术后并发症IV']
# #y_test_data5 = y_test['术后并发症IV']
# svm5 = SVC(kernel='rbf', C=100.0,gamma=100.0, random_state=0)
# svm5.fit(X_data_pca, y_data5)
# #y_pred5 = svm5.predict(X_test_data_pca)
# print('准确率: %.2f' % np.mean(cross_val_score(svm5,X_data_pca,y_data5,cv=3,scoring='recall')))
# #print(y_pred5)
# #print(y_test_data5.real)
# print('------------预测V级(死亡)--------------')
# y_data6 = Y['V级(死亡)']
# #y_test_data6 = y_test['V级(死亡)']
# svm6 = SVC(kernel='rbf', C=100.0,gamma=100.0, random_state=0)
# #svm6.fit(X_data_pca, y_data6)
# #y_pred6 = svm6.predict(X_test_data_pca)
# print('准确率: %.2f' % np.mean(cross_val_score(svm6,X_data_pca,y_data6,cv=3,scoring='recall')))
# #print(y_pred6)
# #print(y_test_data6.real)