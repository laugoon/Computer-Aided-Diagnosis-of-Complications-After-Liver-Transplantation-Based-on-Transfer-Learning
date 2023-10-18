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


label_list=['胸腔积液','凝血功能指标','转氨酶指标','胆红素指标','急性肺损伤','术后出血','术后感染','胆道并发症','原发性移植肝无功能']
method_list=['PCA+SVM','TCA+SVM','HDA+SVM']
source_feature_name = ['年龄','身高', '体重', 'BMI', 'O型', 'A型' ,'B型', 'AB型' ,'乙肝携带' ,
 '术式(经典1背驮0)', '手术时间min' ,'无肝期时间min' ,'热缺血时间min' ,'冷缺血时间min', '红细胞', '血浆', '自体血',
 '4%白蛋白' ,'2%白蛋白', '纯白蛋白g', 'NS' ,'LR', '万汶' ,'佳乐施', '总入量', '出血量', '胸水' ,'腹水', '总尿量',
 '速尿mg', '甘露醇ml', '碳酸氢钠ml', '纤维蛋白原g' ,'凝血酶原复合物U',
 'VII因子' ,'氨甲环酸g/h', '氨甲环酸入壶g' ,'去甲肾上腺素维持' ,'去甲肾上腺素出室', '肾上腺素维持', '肾上腺素出室',
 '多巴胺维持mg/h' ,'多巴胺出室' ,'开放时阿托品' ,'开放时最低心率' ,'开放时最低SBP' ,'开放时最低DBP', '开放时最低MBP',
 '再灌注后综合征', '切脾', '肝肾联合移植' ,'特利加压素ml/h',
 'Hb-pre' ,'HCT-pre', 'MCV-pre', 'MCH-pre' ,'MCHC-pre' ,'RDW-CVO-pre', 'PLT-pre', 'MPV-pre' ,'PDW-pre','LCR-pre', 
 'Hb-post', 'HCT-post', 'MCV-post' ,'MCH-post', 'MCHC-post',
 'RDW-CVO-post', 'PLT-post' ,'MPV-post' ,'PDW-post', 'LCR-post',
# 血常规
 'Hb-2', 'HCT-2' ,'MCV-2' ,'MCH-2', 'MCHC-2',
 'RDW-CVO-2', 'PLT-2', 'MPV-2','PDW-2' ,'LCR-2','Hb-4' ,'HCT-4', 'MCV-4', 'MCH-4', 'MCHC-4',
 'RDW-CVO-4', 'PLT-4' ,'MPV-4', 'PDW-4' ,'LCR-4','Hb-6',
 'HCT-6', 'MCV-6', 'MCH-6' ,'MCHC-6', 'RDW-CVO-6', 'PLT-6', 'MPV-6' ,'PDW-6',
 'LCR-6','Hb-14', 'HCT-14' ,'MCV-14' ,'MCH-14' ,'MCHC-14','RDW-CVO-14', 'PLT-14' ,'MPV-14', 'PDW-14' ,'LCR-14',
#  生化
 'AST-pre', 'ALT-pre','TBIL-pre' ,'ALB-pre', 'BUN-pre', 'Cr-pre' ,'Glu-pre' ,'K-pre' ,'Na-pre','Ca-pre',
 'AST-post', 'ALT-post' ,'TBIL-post', 'ALB-post' ,'BUN-post','Cr-post', 'K-post', 'Na-post' ,'Ca-post',
 'AST-2' ,'ALT-2' ,'TBIL-2' ,'ALB-2','BUN-2', 'Cr-2' ,'K-2' ,'Na-2' ,'Ca-2',
 'AST-4' ,'ALT-4', 'TBIL-4' ,'ALB-4','BUN-4', 'Cr-4', 'K-4' ,'Na-4', 'Ca-4',
'AST-6', 'ALT-6', 'TBIL-6', 'ALB-6','BUN-6','Cr-6', 'K-6' ,'Na-6', 'Ca-6',
'AST-14' ,'ALT-14' ,'TBIL-14' ,'ALB-14','BUN-14' ,'Cr-14', 'K-14', 'Na-14' ,'Ca-14',
# 血气
'PH-pre','PCO2-pre','PO2-pre','Na-pre.1','K-pre.1','Ca-pre.1','Glu-pre.1','Lac-pre','Hct-pre','BE(B)-pre', 'Hb-pre.1',
'PH-0','PCO2-0','PO2-0','Na-0','K-0','Ca-0','Glu-0','Lac-0','Hct-0','BE(B)-0','Hb-0',

'PH-30','PCO2-30','PO2-30','Na-30','K-30','Ca-30','Glu-30','Lac-30','Hct-30','BE(B)-30','Hb-30',

'PH-icu','PCO2-icu','PO2-icu','Na-icu','K-icu','Ca-icu','Glu-icu','Lac-icu','Hct-icu','BE(B)-icu','Hb-icu',
'PH-2d' ,'PCO2-2d' ,'PO2-2d' ,'Na-2d' ,'K-2d', 'Ca-2d' ,'Glu-2d','Lac-2d', 'Hct-2d' ,'BE(B)-2d', 'Hb-2d',

'PA-pre','PT-pre','PR-pre','APTT-pre','FBG-pre','INR-pre' ,
 'PA-post','PT-post','PR-post','APTT-post','FBG-post','INR-post','D-Dimer-post',

'PA-2','PT-2','PR-2','APTT-2','FBG-2','INR-2','D-Dimer-2',
'PA-4' ,'PT-4', 'PR-4', 'APTT-4', 'FBG-4' ,'INR-4', 'D-Dimer-4',
'PA-6', 'PT-6' ,'PR-6' ,'APTT-6','FBG-6', 'INR-6',

'红细胞POD0', '红细胞POD2' , '红细胞POD4' , '红细胞POD6','红细胞POD8' , '红细胞POD10', '红细胞POD12',
'红细胞POD14', '术后红细胞总量',
'血浆POD0' , '血浆POD2' ,'血浆POD4', '血浆POD6' , '血浆POD8','血浆POD10','血浆POD12' , '血浆POD14' , '术后血浆总量',
'血小板POD0' ,'血小板POD2' , '血小板POD4' , '血小板POD6' , '血小板POD8','血小板POD10' , '血小板POD14','血小板POD14+'
]


# 作图部分
def getFig(method_name,feature_order,shap_values,label_name,max_display=10,feature_names=source_feature_name):
    # 图片大小
    plt.figure(figsize=(1.5 * max_display + 1, 0.8 * max_display + 1))
    # 绘制横着的条形图, 分别计算传入的几个参数值, 横着的用height控制线条宽度
    feature_inds = feature_order[:max_display]#最好的几个特征下标
    y_pos = np.arange(len(feature_inds))#各个y的坐标通过等差数组计算
    color='blue'
    num_features = shap_values.shape[0]
    if(feature_names==None):
        feature_names = np.array(['FEATURE'+str(i) for i in range(num_features)])
    plt.barh(y_pos, shap_values[feature_inds], height=0.7, align='center', color=color)
    # 设置轴上刻度
    plt.yticks(y_pos, fontsize=13)# 设置设置轴上刻度
    plt.gca().set_yticklabels([feature_names[i] for i in feature_inds])#坐标轴
    # 设置坐标轴与图片标题
    plt.xlabel('GLOBAL_VALUE', fontsize=13) 
    plt.title('{} {} Feature Importance'.format(method_name,label_name), fontsize=30) 
    plt.savefig("./fig/{}_{}_origin_shap.png".format(method_name,label_name))
    # plt.show()
    

#    计算原特征空间shap 值
def shap_origin_features(method_name='PCA+SVM',label_num=9,max_display = 10,feature_names=source_feature_name):
    # method_name='PCA+SVM'
    # 读取数据:降维后shap值和变换矩阵
    all_labels_shap_values=pd.read_csv('./data/{}_cross_val_shap.csv'.format(method_name))
    X_components=pd.read_csv('./data/{}_components.csv'.format(method_name))
    # 取变换矩阵绝对值, 求每个原特征使用比例
    X_components=np.array(np.maximum(X_components,-X_components))
    component_sum=X_components.sum(axis=1)
    # print("component_sum:\n{}".format(component_sum))
    component_sum=component_sum.reshape((len(component_sum),1))
    use_ratio = X_components / component_sum
    # 计算原特征shap
    shap_origin=np.dot(all_labels_shap_values, use_ratio)
    #保存一下
    shap_origin_df = pd.DataFrame(shap_origin)
    shap_origin_df.columns=feature_names #加上特征名
    shap_origin_df.to_csv('./data/{}_shap_origin.csv'.format(method_name),encoding="utf-8-sig", index=False)
    print('PCA+SVM shap saved')
    # 循环拆分出特定目标label的shap
    for i in range(label_num):
        shap_values=shap_origin[i:i+1]
        # 为作图函数的输入参数匹配形状
        # shap_values=np.array(shap_values).flatten()
        shap_values=np.array(shap_values).flatten()
        # print('shap_values:\n',shap_values)
        # 排序
        feature_order = np.argsort(shap_values)#升序索引
        feature_order = feature_order[-min(max_display, len(feature_order)):]#取最后几个元素(所有特征或最大显示数)

        ######画图######
        getFig(method_name,feature_order,shap_values,label_list[i], max_display=max_display)
        
if __name__ == '__main__':
    shap_origin_features(max_display=10)
    # for i in range(len(method_list)):
    #     shap_after_dimension_reduction(method_name=method_list[i])