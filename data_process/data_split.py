# coding: utf-8
'''
21.12.12修改了划分属性, 对齐两域, 使数据同构
一、对数据划分测试集与验证集

二、这里对处理的数据（测试集），进行源域与目标域的划分：

  医疗数据主要包括：一般情况、术中情况、血常规、血气、生化、凝血、术后输血情况、出院前后转归。
  每个表格中记录不同时间段的医疗指标数据，根据时间段划分源域与目标域


  1、血常规：术前、手术结束、距手术结束1、2、3、4、5、6、7、14
  源域：术前、手术结束，2，4，6，14
  目标域：术前、手术结束，1，3，5，7

  2、生化:术前、手术结束、距手术结束1、2、3、4、5、6、7、14
  源域：术前、手术结束，2，4，6，14
  目标域：术前、手术结束，1，3，5，7

  3、血气：术前、门脉开放时、门脉开放30min、门脉开放60min、门脉开放150min、手术结束、入ICU、门脉开放1d、门脉开放2d
  源域：术前、门脉开放时、门脉开放30min、门脉开放150min、入ICU、门脉开放2d
  目标域：术前、门脉开放时、门脉开放60min、手术结束、门脉开放1d

  4、凝血：术前、手术结束、距手术结束1、2、3、4、5、6、7
  源域：术前、手术结束、距手术结束2、4、6
  目标域：术前、手术结束、距手术结束1、3、5、7

  5、术后输血情况：红细胞、血浆、血小板0--14+
  源域：红细胞、血浆、血小板：0、2、4、6、8、10、12、14
  目标域：红细胞、血浆、血小板：1、3、5、7、9、11、13、14+


'''

import pandas as pd
import numpy as np
from sklearn import preprocessing
from scipy.spatial.distance import pdist
from sklearn.model_selection import train_test_split

# print('-------划分训练集与测试集-----------')

# def data_split(data):

#  data.drop(['序列号', 'Unnamed: 0.1', 'Unnamed: 0'], axis=1, inplace=True)
#  # print(data.columns.values)
#  #使用train_test_split函数划分数据集(训练集占80%，验证集占20%)
#  data, data_test = train_test_split(data, test_size=0.20, random_state=0)
#  print(data.shape)
#  print(data_test.shape)
#  X = data.iloc[:,0:457]
#  Y = data.iloc[:, 457:463]
#  X_test = data_test.iloc[:, 0:457]
#  Y_test = data_test.iloc[:, 457:463]
#  # X = data.iloc[:,0:457]
#  # Y = data.iloc[:,457:463]
#  return X,Y,X_test,Y_test

print('---------划分源域数据------------')
def source_target_split(data):
 source_data = data.loc[:,['年龄','身高', '体重', 'BMI', 'O型', 'A型' ,'B型', 'AB型' ,'乙肝携带' ,
#  source_data = data.loc[:,['年龄','身高', '体重', 'BMI', 'O型', 'A型' ,'B型', 'AB型' ,'RH(阳1阴0)', '乙肝携带' ,
#  '术式(经典1背驮2)', '手术时间min' ,'无肝期时间min' ,'热缺血时间min' ,'冷缺血时间min', '红细胞', '血浆', '自体血',
 '术式(经典1背驮0)', '手术时间min' ,'无肝期时间min' ,'热缺血时间min' ,'冷缺血时间min', '红细胞', '血浆', '自体血',
 '4%白蛋白' ,'2%白蛋白', '纯白蛋白g', 'NS' ,'LR', '万汶' ,'佳乐施', '总入量', '出血量', '胸水' ,'腹水', '总尿量',
#  'I期尿量', 'II期尿量' ,'III期尿量', 
 '速尿mg', '甘露醇ml', '碳酸氢钠ml', '纤维蛋白原g' ,'凝血酶原复合物U',
 'VII因子' ,'氨甲环酸g/h', '氨甲环酸入壶g' ,'去甲肾上腺素维持' ,'去甲肾上腺素出室', '肾上腺素维持', '肾上腺素出室',
 '多巴胺维持mg/h' ,'多巴胺出室' ,'开放时阿托品' ,'开放时最低心率' ,'开放时最低SBP' ,'开放时最低DBP', '开放时最低MBP',
 '再灌注后综合征', '切脾', '肝肾联合移植' ,'特利加压素ml/h',
 # 血常规
#  'Hb-pre_x' ,'HCT-pre', 'MCV-pre', 'MCH-pre' ,'MCHC-pre' ,'RDW-CVO-pre', 'PLT-pre', 'MPV-pre' ,'PDW-pre',
 'Hb-pre' ,'HCT-pre', 'MCV-pre', 'MCH-pre' ,'MCHC-pre' ,'RDW-CVO-pre', 'PLT-pre', 'MPV-pre' ,'PDW-pre','LCR-pre', 
 'Hb-post', 'HCT-post', 'MCV-post' ,'MCH-post', 'MCHC-post',
 'RDW-CVO-post', 'PLT-post' ,'MPV-post' ,'PDW-post', 'LCR-post',

 'Hb-2', 'HCT-2' ,'MCV-2' ,'MCH-2', 'MCHC-2',
 'RDW-CVO-2', 'PLT-2', 'MPV-2','PDW-2' ,'LCR-2','Hb-4' ,'HCT-4', 'MCV-4', 'MCH-4', 'MCHC-4',
 'RDW-CVO-4', 'PLT-4' ,'MPV-4', 'PDW-4' ,'LCR-4','Hb-6',
 'HCT-6', 'MCV-6', 'MCH-6' ,'MCHC-6', 'RDW-CVO-6', 'PLT-6', 'MPV-6' ,'PDW-6',
 'LCR-6','Hb-14', 'HCT-14' ,'MCV-14' ,'MCH-14' ,'MCHC-14','RDW-CVO-14', 'PLT-14' ,'MPV-14', 'PDW-14' ,'LCR-14',
#  生化
#  'AST-pre', 'ALT-pre','TBIL-pre' ,'ALB-pre', 'BUN-pre', 'Cr-pre' ,'Glu-pre_x' ,'K-pre_x' ,'Na-pre_x','Ca-pre_x',
 'AST-pre', 'ALT-pre','TBIL-pre' ,'ALB-pre', 'BUN-pre', 'Cr-pre' ,'Glu-pre' ,'K-pre' ,'Na-pre','Ca-pre',
 'AST-post', 'ALT-post' ,'TBIL-post', 'ALB-post' ,'BUN-post','Cr-post', 'K-post', 'Na-post' ,'Ca-post',
 'AST-2' ,'ALT-2' ,'TBIL-2' ,'ALB-2','BUN-2', 'Cr-2' ,'K-2' ,'Na-2' ,'Ca-2',
 'AST-4' ,'ALT-4', 'TBIL-4' ,'ALB-4','BUN-4', 'Cr-4', 'K-4' ,'Na-4', 'Ca-4',
'AST-6', 'ALT-6', 'TBIL-6', 'ALB-6','BUN-6','Cr-6', 'K-6' ,'Na-6', 'Ca-6',
'AST-14' ,'ALT-14' ,'TBIL-14' ,'ALB-14','BUN-14' ,'Cr-14', 'K-14', 'Na-14' ,'Ca-14',
# 血气
# 'PH-pre','PCO2-pre','PO2-pre','Na-pre_y','K-pre_y','Ca-pre_y','Glu-pre_y','Lac-pre','Hct-pre','BE(B)-pre', 'Hb-pre_y',
'PH-pre','PCO2-pre','PO2-pre','Na-pre.1','K-pre.1','Ca-pre.1','Glu-pre.1','Lac-pre','Hct-pre','BE(B)-pre', 'Hb-pre.1',
'PH-0','PCO2-0','PO2-0','Na-0','K-0','Ca-0','Glu-0','Lac-0','Hct-0','BE(B)-0','Hb-0',

'PH-30','PCO2-30','PO2-30','Na-30','K-30','Ca-30','Glu-30','Lac-30','Hct-30','BE(B)-30','Hb-30',
# 'PH-150','PCO2-150','PO2-150','Na-150','K-150','Ca-150','Glu-150','Lac-150',

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
# '血小板POD0' ,'血小板POD2' , '血小板POD4' , '血小板POD6' , '血小板POD8','血小板POD10' , '血小板POD12', '血小板POD14'
# ,'术后带管时间h', 'ICU驻留时间d', '术后住院时间d', '术后住院转归'
]]

 # source_data_lable = data.loc[:,['术后并发症I','术后并发症II','术后并发症IIIa','术后并发症IIIb',
 #                 '术后并发症IV','V级(死亡)']]


 print('---------划分目标域域数据------------')
 target_data = data.loc[:,['年龄','身高', '体重', 'BMI', 'O型', 'A型' ,'B型', 'AB型' , '乙肝携带' ,
#  target_data = data.loc[:,['年龄','身高', '体重', 'BMI', 'O型', 'A型' ,'B型', 'AB型' ,'RH(阳1阴0)', '乙肝携带' ,
#  '术式(经典1背驮2)', '手术时间min' ,'无肝期时间min' ,'热缺血时间min' ,'冷缺血时间min', '红细胞', '血浆', '自体血',
 '术式(经典1背驮0)', '手术时间min' ,'无肝期时间min' ,'热缺血时间min' ,'冷缺血时间min', '红细胞', '血浆', '自体血',
 '4%白蛋白' ,'2%白蛋白', '纯白蛋白g', 'NS' ,'LR', '万汶' ,'佳乐施', '总入量', '出血量', '胸水' ,'腹水', '总尿量',
#  'I期尿量', 'II期尿量' ,'III期尿量', 
 '速尿mg', '甘露醇ml', '碳酸氢钠ml', '纤维蛋白原g' ,'凝血酶原复合物U',
 'VII因子' ,'氨甲环酸g/h', '氨甲环酸入壶g' ,'去甲肾上腺素维持' ,'去甲肾上腺素出室', '肾上腺素维持', '肾上腺素出室',
 '多巴胺维持mg/h' ,'多巴胺出室' ,'开放时阿托品' ,'开放时最低心率' ,'开放时最低SBP' ,'开放时最低DBP', '开放时最低MBP',
 '再灌注后综合征', '切脾', '肝肾联合移植' ,'特利加压素ml/h',
#  'Hb-pre_x' ,'HCT-pre', 'MCV-pre', 'MCH-pre' ,'MCHC-pre' ,'RDW-CVO-pre', 'PLT-pre', 'MPV-pre' ,'PDW-pre', 'LCR-pre', 
 'Hb-pre' ,'HCT-pre', 'MCV-pre', 'MCH-pre' ,'MCHC-pre' ,'RDW-CVO-pre', 'PLT-pre', 'MPV-pre' ,'PDW-pre','LCR-pre', 
 'Hb-post', 'HCT-post', 'MCV-post' ,'MCH-post', 'MCHC-post',
 'RDW-CVO-post', 'PLT-post' ,'MPV-post' ,'PDW-post', 'LCR-post',
  'Hb-1' ,'HCT-1','MCV-1' ,'MCH-1', 'MCHC-1', 'RDW-CVO-1' ,'PLT-1' ,'MPV-1' ,'PDW-1', 'LCR-1',
 'Hb-3', 'HCT-3', 'MCV-3' ,'MCH-3', 'MCHC-3', 'RDW-CVO-3','PLT-3' ,'MPV-3', 'PDW-3', 'LCR-3',
 'Hb-5', 'HCT-5' ,'MCV-5','MCH-5' ,'MCHC-5', 'RDW-CVO-5', 'PLT-5', 'MPV-5', 'PDW-5', 'LCR-5',
'Hb-7' ,'HCT-7' ,'MCV-7' ,'MCH-7' ,'MCHC-7' ,'RDW-CVO-7', 'PLT-7','MPV-7' ,'PDW-7', 'LCR-7',

# 'AST-pre', 'ALT-pre','TBIL-pre' ,'ALB-pre', 'BUN-pre', 'Cr-pre' ,'Glu-pre_x' ,'K-pre_x' ,'Na-pre_x','Ca-pre_x',
'AST-pre', 'ALT-pre','TBIL-pre' ,'ALB-pre', 'BUN-pre', 'Cr-pre' ,'Glu-pre' ,'K-pre' ,'Na-pre','Ca-pre', 
'AST-post', 'ALT-post' ,'TBIL-post', 'ALB-post' ,'BUN-post','Cr-post', 'K-post', 'Na-post' ,'Ca-post',

'AST-1', 'ALT-1', 'TBIL-1', 'ALB-1','BUN-1' ,'Cr-1' ,'K-1', 'Na-1' ,'Ca-1',
'AST-3' ,'ALT-3' ,'TBIL-3', 'ALB-3','BUN-3' ,'Cr-3' ,'K-3', 'Na-3' ,'Ca-3',
'AST-5', 'ALT-5', 'TBIL-5', 'ALB-5','BUN-5', 'Cr-5' ,'K-5', 'Na-5' ,'Ca-5',
'AST-7', 'ALT-7', 'TBIL-7' ,'ALB-7','BUN-7' ,'Cr-7' ,'K-7', 'Na-7' ,'Ca-7',

# 'PH-pre','PCO2-pre','PO2-pre','Na-pre_y','K-pre_y','Ca-pre_y','Glu-pre_y','Lac-pre','Hct-pre','BE(B)-pre', 'Hb-pre_y',
'PH-pre','PCO2-pre','PO2-pre','Na-pre.1','K-pre.1','Ca-pre.1','Glu-pre.1','Lac-pre','Hct-pre','BE(B)-pre', 'Hb-pre.1',
'PH-0','PCO2-0','PO2-0','Na-0','K-0','Ca-0','Glu-0','Lac-0','Hct-0','BE(B)-0','Hb-0',
'PH-60' ,'PCO2-60', 'PO2-60','Na-60','K-60','Ca-60','Glu-60','Lac-60','Hct-60','BE(B)-60','Hb-60',

'PH-end','PCO2-end','PO2-end','Na-end','K-end','Ca-end','Glu-end','Lac-end','Hct-end','BE(B)-end','Hb-end',
'PH-1d' ,'PCO2-1d','PO2-1d', 'Na-1d' ,'K-1d' ,'Ca-1d', 'Glu-1d', 'Lac-1d', 'Hct-1d', 'BE(B)-1d','Hb-1d' ,

'PA-pre','PT-pre','PR-pre','APTT-pre','FBG-pre','INR-pre' ,
 'PA-post','PT-post','PR-post','APTT-post','FBG-post','INR-post','D-Dimer-post',

'PA-1','PT-1','PR-1','APTT-1','FBG-1','INR-1','D-Dimer-1',
'PA-3','PT-3', 'PR-3', 'APTT-3' ,'FBG-3' ,'INR-3', 'D-Dimer-3',
# 'PA-5', 'PT-5','PR-5' ,'APTT-5' ,'FBG-5' ,'INR-5' ,'D-Dimer-5',
'PA-7', 'PT-7' ,'PR-7' ,'APTT-7' ,'FBG-7' ,'INR-7',

'红细胞POD1','红细胞POD3','红细胞POD5','红细胞POD7','红细胞POD9','红细胞POD11','红细胞POD13','红细胞POD14+','术后红细胞总量',
'血浆POD1','血浆POD3','血浆POD5','血浆POD7','血浆POD9','血浆POD11','血浆POD13','血浆POD14+', '术后血浆总量',
 '血小板POD1','血小板POD3','血小板POD5','血小板POD7','血小板POD9','血小板POD11','血小板POD13','血小板POD14+'
# ,'术后带管时间h', 'ICU驻留时间d', '术后住院时间d', '术后住院转归'
]]


 # target_data_lable = data.loc[:,['术后并发症I','术后并发症II','术后并发症IIIa','术后并发症IIIb',
 #                 '术后并发症IV','V级(死亡)'] ]

 # print('source_data的shape为：')
 # print(source_data.shape)
 # print('target_data的shape为：')
 # print(target_data.shape)



 return  source_data,target_data

# def label_split(data):
#   Pleural_Effusion_label= data.loc[:,['所有胸腔积液情况'] ]
#   Pleural_Effusion_label.to_csv('./data/Pleural_Effusion_label.csv')
#   return Pleural_Effusion_label

if __name__ == '__main__':
  data = pd.read_csv('./data/fill_NA_features.csv')
  # labels = pd.read_csv('./data/labels.csv')
  labels = pd.read_csv('./data/fill_labels.csv')
  # print('data:',data['Unnamed: 0.1'])
  # data.drop(['序列号'], axis=1, inplace=True)
  print('data:',data)
  source_data,target_data = source_target_split(data)
  print('———————————————————将源域、目标域数据进行保存————————————————————————')
  source_data.to_csv('./data/source_data.csv', index=False)
  target_data.to_csv('./data/target_data.csv', index=False)
  #  labels=label_split(data)
  #X,y,X_test,y_test = data_split(data)

  
  #  print(X.shape)
  #  print(y.shape)
  #  print(X_test.shape)
  #  print(y_test.shape)

  print("shape of source_data: ",source_data.shape)
  print("shape of target_data: ",target_data.shape)
  print("shape of label: ",labels.shape)