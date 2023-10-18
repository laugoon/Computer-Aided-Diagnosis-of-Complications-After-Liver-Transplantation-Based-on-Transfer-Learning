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


# label_list=['胸腔积液','凝血功能指标','转氨酶指标','胆红素指标','急性肺损伤','术后出血','术后感染','胆道并发症','原发性移植肝无功能']
label_list=['Pleural Effusion','High INR','High ALT or AST','High TBIL','Pneumonia','Postoperative Bleeding','Postoperative Infection','Biliary Complications','PNF']


method_list=['PCA+SVM','TCA+SVM','HDA+SVM']
wrapper_method_list=['PCA+SVM','TCA+SVM']
# Recall_scores_base=np.array([1.00,0.88,0.74,0.79,1.00,0.78])
# F1_scores_base=np.array([0.9804560260586319,0.9354838709677419,0.8493150684931507,0.8813559322033898,0.9895104895104895,0.8732394366197184])
# Accuracy_scores=np.array([1.00,0.86,0.79,0.79,1.00,0.80])
# F1_scores=np.array([0.9868852459016394,0.9239130434782609,0.88,0.8813559322033898,0.9895104895104895,0.888888888888889])



# 作图部分
def getFig(method_name,feature_order,shap_values,label_name,feature_names=None,max_display=10,wrapper_flag=False):
    # 图片大小
    plt.figure(figsize=(1.2 * max_display + 1, 0.4* max_display + 1))
    # 绘制横着的条形图, 分别计算传入的几个参数值, 横着的用height控制线条宽度
    feature_inds = feature_order[:max_display]#最好的几个特征下标
    # y_values=shap_values[feature_inds]#最好特征的shap值
    y_values=[number for number in shap_values[feature_inds]]#用特征ID取出对应SHAP值
    num_features = shap_values.shape[0]#总特征个数
    #所有特征名-异常值处理
    if(feature_names is None):#所有特征名, 如果没有给定文字, 就用默认的
        feature_names = np.array(['FEATURE'+str(i) for i in range(num_features)])
    # 最好特征名
    y_names=[feature_names[i] for i in feature_inds]
    
    y_pos = np.arange(len(feature_inds))#各个y的坐标通过等差数组计算
    color='blue'

    bh=plt.barh(y_pos, y_values, height=0.7, align='center', color=color)
    

    # 设置轴上刻度
    plt.yticks(y_pos, fontsize=22)# 设置设置轴上刻度
    plt.gca().set_yticklabels([feature_names[i] for i in feature_inds])#坐标轴
    # 设置坐标轴与图片标题
    y_max=y_values[max_display-1]#shap中的最大值
    plt.xlim(0, y_max*1.13)#横坐标范围设置
    plt.xticks(fontsize=20)# 设置设置轴上刻度
    ###设置坐标轴的粗细
    ax=plt.gca();#获得坐标轴的句柄
    ax.spines['bottom'].set_linewidth(2);###设置底部坐标轴的粗细
    ax.spines['left'].set_linewidth(2);####设置左边坐标轴的粗细
    ax.spines['right'].set_linewidth(2);###设置右边坐标轴的粗细
    ax.spines['top'].set_linewidth(2);####设置上部坐标轴的粗细

    plt.xlabel('Global Shapley Value', fontsize=22) 
    # plt.title('{} {} Feature Importance'.format(method_name,label_name), fontsize=40) 
    # plt.title('{} Feature Importance'.format(label_name), fontsize=20) 
    # 给条形图添加数据标注
    for rect in bh:
        w=rect.get_width()
        plt.text(w, rect.get_y()+0.3, " %.3f" %w, fontsize=20)#设置位置-横,纵. 显示文字
    
    plt.subplots_adjust(left=0.45,bottom=0.13,top=0.99,right=0.99)#调整 图表 的上下左右
    # #保存png图片
    # if(wrapper_flag):
    #     plt.savefig("./fig/wrapper/{}/{}_{}_wrapper_shap.png".format(method_name,method_name,label_name))
    # else:
    #     plt.savefig("./fig/{}_{}_shap.png".format(method_name,label_name))
    
    # 保存pdf图片
    if(wrapper_flag):
        plt.savefig("./fig/wrapper/{}/{}_{}_wrapper_shap.pdf".format(method_name,method_name,label_name))
    else:
        plt.savefig("./fig/{}_{}_shap.pdf".format(method_name,label_name))
    
    # plt.show()

# 作图部分版本2-更换图样式
def getFigV2(method_name,feature_order,shap_values,label_name,feature_names=None,max_display=10,wrapper_flag=False):
    feature_inds = feature_order[:max_display]#最好的几个特征下标
    # y_values=shap_values[feature_inds]#最好特征的shap值
    y_values=[number for number in shap_values[feature_inds]]#用特征ID取出对应SHAP值
    num_features = shap_values.shape[0]#总特征个数
    #所有特征名-异常值处理
    if(feature_names is None):#所有特征名, 如果没有给定文字, 就用默认的
        feature_names = np.array(['FEATURE'+str(i) for i in range(num_features)])
    # 最好特征名
    y_names=[feature_names[i] for i in feature_inds]
    #shap中的最大值
    y_max=y_values[max_display-1]
    
    # 图片设置
    # 图片大小
    plt.figure(figsize=(0.9 * max_display + 1, 0.8* max_display + 1))
    # 创建条形图
    plt.bar(y_names, y_values,color='darkgreen')


    # 添加数值标签
    for i, v in enumerate(y_values):
        plt.text(i, v+0.02*y_max , str("%.3f" %v), ha='center')
    
    plt.subplots_adjust(bottom=0.3)#调整 图表 的上下左右
    # 添加y轴标签
    plt.ylabel('Importance Value')
    plt.xticks(rotation=90)


    # # 图片大小
    # plt.figure(figsize=(0.8 * max_display + 1, 0.4* max_display + 1))
    # # 绘制横着的条形图, 分别计算传入的几个参数值, 横着的用height控制线条宽度

    # # feature_inds = feature_order[:max_display]#最好的几个特征下标
    # # # y_values=shap_values[feature_inds]#最好特征的shap值
    # # y_values=[number for number in shap_values[feature_inds]]#用特征ID取出对应SHAP值
    # # num_features = shap_values.shape[0]#总特征个数
    # # #所有特征名-异常值处理
    # # if(feature_names is None):#所有特征名, 如果没有给定文字, 就用默认的
    # #     feature_names = np.array(['FEATURE'+str(i) for i in range(num_features)])
    # # # 最好特征名
    # # y_names=[feature_names[i] for i in feature_inds]
    
    # y_pos = np.arange(len(feature_inds))#各个y的坐标通过等差数组计算
    # color='blue'
    # bh=plt.barh(y_pos, y_values, height=0.7, align='center', color=color)
    

    # # 设置轴上刻度
    # plt.yticks(y_pos, fontsize=15)# 设置设置轴上刻度
    # plt.gca().set_yticklabels(y_names)#坐标轴
    # # 设置坐标轴与图片标题
    # y_max=y_values[max_display-1]#shap中的最大值
    # plt.xlim(0, y_max*1.13)#横坐标范围设置
    # plt.xlabel('Global_Shapley_Value', fontsize=18) 
    # # plt.title('{} {} Feature Importance'.format(method_name,label_name), fontsize=40) 
    # plt.title('{} Feature Importance'.format(label_name), fontsize=20) 
    # # 给条形图添加数据标注
    # for rect in bh:
    #     w=rect.get_width()
    #     plt.text(w, rect.get_y()+0.3, " %.3f" %w)#设置位置-横,纵. 显示文字
    
    # plt.subplots_adjust(left=0.43)#调整 图表 的上下左右
    # # if(wrapper_flag):
    # #     plt.savefig("./fig/wrapper/{}/{}_{}_wrapper_shap.png".format(method_name,method_name,label_name))
    # # else:
    # #     plt.savefig("./fig/{}_{}_shap.png".format(method_name,label_name))
    if(wrapper_flag):
        plt.savefig("./fig/v2/wrapper/{}/{}_{}_wrapper_shap_v2.png".format(method_name,method_name,label_name))
    else:
        plt.savefig("./fig/v2/{}_{}_shap_v2.png".format(method_name,label_name))
    # plt.show()
    
    

#    shap 值条形图-降维后还原
def shap_after_dimension_reduction(method_name='PCA+SVM',label_num=9,max_display = 10):
    # method_name='PCA+SVM'
    all_labels_shap_values=pd.read_csv('./data/{}_cross_val_shap.csv'.format(method_name))
    # 循环拆分出特定目标label的shap
    for i in range(label_num):
        shap_values=all_labels_shap_values[i:i+1]
        # 为作图函数的输入参数匹配形状
        # shap_values=np.array(shap_values).flatten()
        shap_values=np.array(shap_values).flatten()
        # print('shap_values:\n',shap_values)
        # 排序
        feature_order = np.argsort(shap_values)#升序索引
        feature_order = feature_order[-min(max_display, len(feature_order)):]#取最后几个元素(所有特征或最大显示数)

        ######画图######
        getFig(method_name,feature_order,shap_values,label_list[i])
        # # 图片大小
        # plt.figure(figsize=(1.5 * max_display + 1, 0.8 * max_display + 1))
        # # 绘制横着的条形图, 分别计算传入的几个参数值, 横着的用height控制线条宽度
        # feature_inds = feature_order[:max_display]#最好的几个特征下标
        # y_pos = np.arange(len(feature_inds))#各个y的坐标通过等差数组计算
        # color='blue'
        # num_features = shap_values.shape[0]
        # feature_names = np.array(['FEATURE'+str(i) for i in range(num_features)])
        # plt.barh(y_pos, shap_values[feature_inds], height=0.7, align='center', color=color)
        # # 设置轴上刻度
        # plt.yticks(y_pos, fontsize=13)# 设置设置轴上刻度
        # plt.gca().set_yticklabels([feature_names[i] for i in feature_inds])#坐标轴
        # # 设置坐标轴与图片标题
        # plt.xlabel('GLOBAL_VALUE', fontsize=13) 
        # plt.title('{} {} Feature Importance'.format(method_name,label_list[i]), fontsize=30) 
        # plt.savefig("./fig/{}_{}_shap.png".format(method_name,label_list[i]))
        # # plt.show()

#    shap 值条形图-wrapper
def shap_wrapper(method_name='PCA+SVM',label_num=9,max_display = 10):
    # method_name='PCA+SVM'
    all_labels_shap_values=pd.read_csv('./data/wrapper_experiment/{}_wrapper_shap.csv'.format(method_name))
    # 循环拆分出特定目标label的shap
    for i in range(label_num):
        shap_values=all_labels_shap_values[i:i+1]
        # 为作图函数的输入参数匹配形状
        # shap_values=np.array(shap_values).flatten()
        shap_values=np.array(shap_values).flatten()
        # print('shap_values:\n',shap_values)
        # 排序
        feature_order = np.argsort(shap_values)#升序索引
        top_features = feature_order[-min(max_display, len(feature_order)):]#取最后几个元素(所有特征或最大显示数)
        # pd.DataFrame(top_features).to_csv('./data/wrapper_experiment/{}_TCA+SVM_top_features.csv'.format(label_list[i]), index=False)

        ######画图######

        # feature_names=np.loadtxt('./data/wrapper_experiment/{}_features_name.txt'.format(method_name),dtype='str')
        feature_names=np.loadtxt('./data/wrapper_experiment/{}_features_name.txt'.format(method_name),dtype='str', delimiter='\n')
        getFig(method_name,top_features,shap_values,label_list[i],wrapper_flag=True,feature_names=feature_names)
        # #换形式
        # getFigV2(method_name,top_features,shap_values,label_list[i],wrapper_flag=True,feature_names=feature_names)
        # if(method_name=='PCA+SVM'):
        #     feature_names=np.loadtxt('./data/wrapper_experiment/important_features_name.txt',dtype='str')
        #     getFig(method_name,feature_order,shap_values,label_list[i],wrapper_flag=True,feature_names=feature_names)
  
def local_shap_wrapper(method_name='TCA+SVM',label_num=9,max_display = 10):
    # method_name='PCA+SVM'
    all_labels_shap_values=pd.read_csv('./data/wrapper_experiment/{}_wrapper_shap.csv'.format(method_name))
    top_features=np.empty(shape=(label_num,max_display))
    

    # 循环拆分出特定目标label的shap
    for i in range(label_num):
        shap_values=all_labels_shap_values[i:i+1]
        # 为作图函数的输入参数匹配形状
        # shap_values=np.array(shap_values).flatten()
        shap_values=np.array(shap_values).flatten()
        print('shap_values shape:\n',shap_values.shape)
        # 排序
        feature_order = np.argsort(shap_values)#升序索引
        feature_order = feature_order[-min(max_display, len(feature_order)):]#取最后几个元素(所有特征或最大显示数)
        top_features[i]=feature_order
 
        # feature_order = np.argsort(shap_values)#升序索引
        # feature_order = feature_order[-min(max_display, len(feature_order)):]#取最后几个元素(所有特征或最大显示数)

        ######画图######

        # feature_names=np.loadtxt('./data/wrapper_experiment/{}_features_name.txt'.format(method_name),dtype='str')
        feature_names=np.loadtxt('./data/wrapper_experiment/{}_features_name.txt'.format(method_name),dtype='str', delimiter='\n')
        getFig(method_name,feature_order,shap_values,label_list[i],wrapper_flag=True,feature_names=feature_names)
        # if(method_name=='PCA+SVM'):
        #     feature_names=np.loadtxt('./data/wrapper_experiment/important_features_name.txt',dtype='str')
        #     getFig(method_name,feature_order,shap_values,label_list[i],wrapper_flag=True,feature_names=feature_names)

    pd.DataFrame(top_features).to_csv('./data/wrapper_experiment/{}_top_features.csv'.format(method_name), index=False)

if __name__ == '__main__':
    # debug
    feature_names=np.loadtxt('./data/wrapper_experiment/{}_features_name.txt'.format("TCA+SVM"),dtype='str', delimiter='\n')
    print(feature_names.shape,"\n",feature_names)
    # debug end

    # comparison_TCA_cross_val()
    # # shap 值条形图-降维后还原
    # for i in range(len(method_list)):
    #     shap_after_dimension_reduction(method_name=method_list[i])

#    shap 值条形图-wrapper
    # for i in range(len(wrapper_method_list)):
    #     shap_wrapper(method_name=wrapper_method_list[i])
    shap_wrapper(method_name=wrapper_method_list[1])


# #    局部shap 值蜂群图-wrapper
#     # 记录最重要特征
#     local_shap_wrapper(method_name=wrapper_method_list[1])    