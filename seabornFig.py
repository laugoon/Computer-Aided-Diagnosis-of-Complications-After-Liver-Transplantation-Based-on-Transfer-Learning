import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

# 查询当前系统所有字体
# from matplotlib.font_manager import FontManager
# import subprocess

# mpl_fonts = set(f.name for f in FontManager().ttflist)

# print('all font list get from matplotlib.font_manager:')
# for f in sorted(mpl_fonts):
#     print('\t' + f)

plt.rc("font",family='KaiTi')


label_list=['术后并发症I','术后并发症II','术后并发症IIIa','术后并发症IIIb','术后并发症IV','V级(死亡)']
# Recall_scores_base=np.array([1.00,0.88,0.74,0.79,1.00,0.78])
# F1_scores_base=np.array([0.9804560260586319,0.9354838709677419,0.8493150684931507,0.8813559322033898,0.9895104895104895,0.8732394366197184])
# Accuracy_scores=np.array([1.00,0.86,0.79,0.79,1.00,0.80])
# F1_scores=np.array([0.9868852459016394,0.9239130434782609,0.88,0.8813559322033898,0.9895104895104895,0.888888888888889])



# regressor_list,scores_base,scores
def getFig(scores_base,scores,fig_name,y_axis_name,now_label,base_label='TCA+SVM',regressor_list=label_list):
    sns.set(font="KaiTi",style="white", font_scale=1.5)
    width = 0.4
    fig, ax = plt.subplots(figsize=(10, 6))
    # ax.bar(regressor_list, scores_base, width, label='Original Features')
    ax.bar(regressor_list, scores_base, width, label=base_label)
    difference = scores - scores_base
    print(np.where(difference > 0, 'g', 'y'))
    # ax.bar(regressor_list, difference, width, bottom=scores_base,
    #     label='Constructed Features',
    #     color=np.where(difference > 0, 'r', 'y'))
    ax.bar(regressor_list, difference, width, bottom=scores_base,
        label=now_label,
        color=np.where(difference > 0, 'r', 'y'))
    ax.set_ylabel(y_axis_name)
    # ax.set_title('Effect of Feature Construction')
    ax.set_title(fig_name)
    ax.legend(loc=3)
    plt.ylim((0, 1))
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("./fig/{}.png".format(fig_name))
    plt.show()
# TCA+EF与TCA对比
def comparison_TCA_EF():
    Recall_scores_base=pd.read_csv('./data/TCA+SVM_recall.csv',index_col=0)
    F1_scores_base=pd.read_csv('./data/TCA+SVM_f1.csv',index_col=0)
    all_dataset_recall = pd.read_csv('./data/EF_afterTCA_recall.csv',index_col=0)
    all_dataset_f1 = pd.read_csv('./data/EF_afterTCA_f1.csv',index_col=0)
    # # 单独EF对比
    # all_dataset_recall = pd.read_csv('./data/EF_recall.csv')
    # all_dataset_f1 = pd.read_csv('./data/EF_f1.csv')
    Recall_scores_base=np.array(Recall_scores_base).flatten()
    F1_scores_base=np.array(F1_scores_base).flatten()
    
    all_dataset_recall=np.array(all_dataset_recall)
    all_dataset_f1=np.array(all_dataset_f1)
    print('Recall_scores_base:\n',Recall_scores_base)
    
    # np.array
    for dataset_idx in range(6):
        getFig(Recall_scores_base,all_dataset_recall[dataset_idx],fig_name='对{}EF'.format(label_list[dataset_idx]),y_axis_name='Recall', now_label='TCA+EF+SVM')
        getFig(F1_scores_base,all_dataset_f1[dataset_idx],fig_name='对{}EF'.format(label_list[dataset_idx]),y_axis_name='F1 Score', now_label='TCA+EF+SVM')
# TCA+随机过采样对比
def comparison_TCA_ROS():
    recall_base=pd.read_csv('./data/TCA+SVM_recall.csv',index_col=0)
    f1_base=pd.read_csv('./data/TCA+SVM_f1.csv',index_col=0)
    recall_oversampling = pd.read_csv('./data/TCA+RandomOverSampling+SVM_recall.csv',index_col=0)
    f1_oversampling = pd.read_csv('./data/TCA+RandomOverSampling+SVM_f1.csv',index_col=0)
    # 为作图函数的输入参数改变形状
    recall_base=np.array(recall_base).flatten()
    f1_base=np.array(f1_base).flatten()
    
    recall_oversampling=np.array(recall_oversampling).flatten()
    f1_oversampling=np.array(f1_oversampling).flatten()
    # print('Recall_scores_base:\n',Recall_scores_base)

    getFig(recall_base,recall_oversampling,fig_name='随机过采样Recall对比图',y_axis_name='Recall', now_label='TCA+ROS+SVM')
    getFig(f1_base,f1_oversampling,fig_name='随机过采样F1对比图',y_axis_name='F1 Score', now_label='TCA+ROS+SVM')
# TCA+SMOTE过采样对比
def comparison_TCA_SMOTE():
    recall_base=pd.read_csv('./data/TCA+SVM_recall.csv',index_col=0)
    f1_base=pd.read_csv('./data/TCA+SVM_f1.csv',index_col=0)
    recall_oversampling = pd.read_csv('./data/TCA+SMOTE+SVM_recall.csv',index_col=0)
    f1_oversampling = pd.read_csv('./data/TCA+SMOTE+SVM_f1.csv',index_col=0)
    # 为作图函数的输入参数改变形状
    recall_base=np.array(recall_base).flatten()
    f1_base=np.array(f1_base).flatten()
    
    recall_oversampling=np.array(recall_oversampling).flatten()
    f1_oversampling=np.array(f1_oversampling).flatten()
    # print('Recall_scores_base:\n',Recall_scores_base)

    getFig(recall_base,recall_oversampling,fig_name='SMOTE过采样Recall对比图',y_axis_name='Recall', now_label='TCA+SMOTE+SVM')
    getFig(f1_base,f1_oversampling,fig_name='SMOTE过采样F1对比图',y_axis_name='F1 Score', now_label='TCA+SMOTE+SVM')
# TCA+ADASYN过采样对比
def comparison_TCA_ADASYN():
    recall_base=pd.read_csv('./data/TCA+SVM_recall.csv',index_col=0)
    f1_base=pd.read_csv('./data/TCA+SVM_f1.csv',index_col=0)
    recall_oversampling = pd.read_csv('./data/TCA+ADASYN+SVM_recall.csv',index_col=0)
    f1_oversampling = pd.read_csv('./data/TCA+ADASYN+SVM_f1.csv',index_col=0)
    # 为作图函数的输入参数改变形状
    recall_base=np.array(recall_base).flatten()
    f1_base=np.array(f1_base).flatten()
    recall_oversampling=np.array(recall_oversampling).flatten()
    f1_oversampling=np.array(f1_oversampling).flatten()
    # print('Recall_scores_base:\n',Recall_scores_base)

    getFig(recall_base,recall_oversampling,fig_name='ADASYN过采样Recall对比图',y_axis_name='Recall', now_label='TCA+ADASYN+SVM')
    getFig(f1_base,f1_oversampling,fig_name='ADASYN过采样F1对比图',y_axis_name='F1 Score', now_label='TCA+ADASYN+SVM')

# TCA+交叉验证对比
def comparison_TCA_cross_val():
    recall_base=pd.read_csv('./data/TCA+SVM_recall.csv',index_col=0)
    f1_base=pd.read_csv('./data/TCA+SVM_f1.csv',index_col=0)
    recall_cross_val = pd.read_csv('./data/TCA+SVM_cross_val_recall.csv',index_col=0)
    f1_cross_val = pd.read_csv('./data/TCA+SVM_cross_val_f1.csv',index_col=0)
    # 为作图函数的输入参数匹配形状
    recall_base=np.array(recall_base).flatten()
    f1_base=np.array(f1_base).flatten()
    recall_cross_val=np.array(recall_cross_val).flatten()
    f1_cross_val=np.array(f1_cross_val).flatten()
    # print('Recall_scores_base:\n',Recall_scores_base)

    getFig(recall_base,recall_cross_val,fig_name='TCA+SVM交叉验证Recall对比图',y_axis_name='Recall', now_label='TCA+SVM交叉验证',base_label='TCA+SVM目标域作为测试集')
    getFig(f1_base,f1_cross_val,fig_name='TCA+SVM交叉验证F1对比图',y_axis_name='F1 Score', now_label='TCA+SVM交叉验证',base_label='TCA+SVM目标域作为测试集')
    
if __name__ == '__main__':
    # comparison_TCA_EF()
    # comparison_TCA_ROS()
    # comparison_TCA_SMOTE()
    # comparison_TCA_ADASYN()
    comparison_TCA_cross_val()