# coding: utf-8
'''
整体思路是从xlsx文件的10个table中合并每个病人的术前、术后信息。   保存到csv文件中去....
'''
from numpy import True_
import pandas as pd
import matplotlib.pyplot as plt
import xlrd
import csv
from pandas.core.frame import DataFrame
import re



# feature_file ='./data/原始病历数据/肝移植回顾数据-2016 -2019.xlsx'
# label_file ='./data/原始病历数据/12.14分类目标.xlsx'
feature_file =r'./data/原始病历数据/肝移植回顾数据-2016 -2019.xlsx'
label_file =r'./data/原始病历数据/12.14分类目标.xlsx'
label_name_in_sheet=['所有胸腔积液情况','INR≥1.6','ALT或AST≥2000IU/mL','TBIL≥10 mg/Dl','肺炎','术后出血整体情况','术后感染整体情况','所有胆漏情况','初期移植肝无功能']
label_name=['胸腔积液','凝血功能指标','转氨酶指标','胆红素指标','急性肺损伤','术后出血','术后感染','胆道并发症','原发性移植肝无功能']

table1=pd.read_excel(feature_file,sheet_name='一般情况')
table2=pd.read_excel(feature_file,sheet_name='术中情况')
table3=pd.read_excel(feature_file,sheet_name='血常规')
table4=pd.read_excel(feature_file,sheet_name='生化')
table5=pd.read_excel(feature_file,sheet_name='血气')
table6=pd.read_excel(feature_file,sheet_name='凝血')
table7=pd.read_excel(feature_file,sheet_name='术后输血情况')
table8=pd.read_excel(feature_file,sheet_name='出院前术后转归')
table9=pd.read_excel(feature_file,sheet_name='总体转归')###为了减少维度，这个表可以先不用
table10=pd.read_excel(feature_file,sheet_name='术中化验')###只有20多条记录，暂不计算这个
# table12=pd.read_excel(feature_file,sheet_name='胸腔积液')
print('features:\n',table1)
# label1=pd.read_excel(label_file,sheet_name='胸腔积液')
# label2=pd.read_excel(label_file,sheet_name='INR')
# label3=pd.read_excel(label_file,sheet_name='转氨酶')
# label4=pd.read_excel(label_file,sheet_name='胆红素')
# label5=pd.read_excel(label_file,sheet_name='肺炎')
# label6=pd.read_excel(label_file,sheet_name='术后出血')
# label7=pd.read_excel(label_file,sheet_name='术后感染')
# label8=pd.read_excel(label_file,sheet_name='胆漏')
# label9=pd.read_excel(label_file,sheet_name='PNF')

data_xlsx = pd.ExcelFile(label_file)
# print("sheet name:",data_xlsx.sheet_names)
labels=pd.DataFrame()
for idx, name in enumerate(data_xlsx.sheet_names):
    df=data_xlsx.parse(sheet_name=name,usecols=[label_name_in_sheet[idx]])
    labels=pd.concat([labels,df],axis=1)
# print('labels:\n',labels)
labels.columns = label_name
labels.to_csv('./data/labels.csv', index=False)


'''
table合并前，先针对每个table进行简单的数据处理：
如初始的特征手动筛选，先删除table中冗余的列；改变可识别的数据类型
'''
print('———————针对每个table进行简单的数据处理————————')

print('——————— 处理table1一般情况————————')
cols = ['序列号','年龄','身高','体重','BMI','O型','A型','B型','AB型','RH(阳1阴0)','乙肝携带']
table1 = table1[cols]
#print(table1.shape)
# table1.drop(['病案号','血型','诊断','并发症','具体情况','间隔时间','血型',
#              '18其它具体情况'], axis = 1,inplace=True)
# # print(len(table1.columns))
# # print(table1.columns.values)
#
# table1.rename(columns={'1乙肝肝硬化':'诊断_1乙肝肝硬化','2乙肝肝硬化肝癌':'诊断_2乙肝肝硬化肝癌',
#                        '3丙肝肝硬化':'诊断_3丙肝肝硬化','4丙肝肝硬化肝癌':'诊断_4丙肝肝硬化肝癌',
#                        '5肝癌':'诊断_5肝癌','6酒精性肝硬化':'诊断_6酒精性肝硬化',
#                        '7酒精性肝硬化肝癌':'诊断_7酒精性肝硬化肝癌','8酒精性肝硬化+乙肝':'8酒精性肝硬化+乙肝',
#                        '9酒精性肝硬化+丙肝':'诊断_9酒精性肝硬化+丙肝','10原发性胆汁性肝硬化':'诊断_10原发性胆汁性肝硬化',
#                        '11自免肝':'诊断_11自免肝','12自免肝+肝癌':'诊断_12自免肝+肝癌',
#                        '13药物性肝衰':'诊断_13药物性肝衰','14其它':'诊断_14其它',
#                        '15胆管癌':'诊断_15胆管癌',
#                        '1门脉血栓':'并发症_1门脉血栓','2门脉高压':'并发症_2门脉高压',
#                        '3食管胃底静脉曲张': '并发症_3食管胃底静脉曲张', '4高血压': '并发症_4高血压',
#                        '5糖尿病': '并发症_5糖尿病', '6腹水': '并发症_6腹水',
#                        '7胸水': '并发症_7胸水', '8肺部感染': '并发症_8肺部感染',
#                        '9腹腔感染': '并发症_9腹腔感染', '10肝肾综合征': '并发症_10肝肾综合征',
#                        '11肝昏迷': '并发症_11肝昏迷', '12上消化道出血': '并发症_12上消化道出血',
#                        '13脾大': '并发症_13脾大', '14脾亢': '并发症_14脾亢',
#                        '15脾切除术':'并发症_15脾切除术','16上腹部手术史':'并发症_16上腹部手术史',
#                        '17二次肝移植':'并发症_17二次肝移植  ','18其它':'并发症_18其它',
#                        },inplace=True)
# # print(table1)
# # print(table1.loc['并发症_15脾切除术'])
# # # a = table1.loc['序列号','并发症_18其它']
# # # print(a)
# # # for i in table1.序列号:
# # #     a = table1.at[i,'并发症_18其它']
# # #     print(a)
# # #     if a==0:
# # #         print("00")
# # #     else:
# # #         a=1
#
#
# #‘暂且不考虑术前并发症因素的影响等一些特征，因此手动先删除这部分特征’
#
# table1.drop(['并发症_1门脉血栓','并发症_2门脉高压', '并发症_3食管胃底静脉曲张','并发症_4高血压',
#             '并发症_5糖尿病','并发症_6腹水','并发症_7胸水','并发症_8肺部感染',
#              '并发症_9腹腔感染','并发症_10肝肾综合征','并发症_11肝昏迷','并发症_12上消化道出血',
#              '并发症_13脾大', '并发症_14脾亢', '并发症_15脾切除术','并发症_16上腹部手术史',
#              '并发症_17二次肝移植  ','并发症_18其它'], axis=1,inplace=True)
# #print(table1)
# #print(table1.columns.values)


print('——————— 处理table2术中情况 ————————')

table2.drop( ['手术开始日期','手术结束日期','门脉阻断时间','门脉开放时间','其它液体'],axis = 1,inplace=True)


#单位换算

#冷缺血时间h----->冷缺血时间min

table2.loc[:,'冷缺血时间h']=table2.loc[:,'冷缺血时间h']*60
table2.rename(columns={'冷缺血时间h':'冷缺血时间min'},inplace=True)
print(table2.loc[:,'冷缺血时间min'])

#特利加压素:ug/h----->ml/h
# 有些特征数值后还加了'ug/h'

val = list(table2['特利加压素ml/h'])
#print('VAL',val)
values = []
for i in range(len(val)):
    if 'ug/h' in str(val[i]):
        #print('ii',val[i])
        a = val[i]
        b = a.split('u',1)[0]
        b = int(b) * (1e-06) / 1.46
        val[i]=b


    values.append(val[i])


table2['new']=values
table2.drop( ['特利加压素ml/h'],axis = 1,inplace=True)
table2.rename(columns={'new':'特利加压素ml/h'},inplace=True)

pd.DataFrame(table2).to_excel('./data/术中情况.xlsx', sheet_name='术中情况', header=True, index=False)
table2=pd.read_excel('./data/术中情况.xlsx',sheet_name='术中情况')

#print('values', values)
#print(table2)





print('———————处理table3血常规————————')
table3.drop( ['术后条码','POD1条码','POD2条码','POD3条码','POD4条码','POD5条码','POD6条码'
            ,'POD7条码','POD14条码'],axis = 1,inplace=True)
# print("tabel3:\n",table3.columns.values)

print('———————处理table4生化————————')
table4.drop( ['术后条码','POD1条码','POD2条码','POD3条码','POD4条码','POD5条码','POD6条码'
            ,'POD7条码','POD14条码'
              ],axis = 1,inplace=True)
#print(table4.columns.values)


print('——————— 处理table5血气————————')
table5.drop( ['姓名','门脉开放时间','术前时间点','开放前时间点','距开放时间-0(min)',
              '开放10min内','距开放时间-10(min)','开放30min内','距开放时间-30(min)'
            ,'开放31-60min','距开放时间-60(min)','开放61-90min内','距开放时间-90(min)',
              '开放91-120min内','距开放时间-120(min)','开放121-150min内','距开放时间-150(min)',
              '开放151-180min内','距开放时间-180(min)','开放181-210min内','距开放时间-210(min)',
              '开放211-240min内','距开放时间-240(min)','术毕','距开放时间-end(min)','入ICU',
              '距开放时间-icu(min)','术后1天','距开放时间-1d(min)','距开放时间-1d(h)',
              '术后2天','距开放时间-2d(min)','距开放时间-2d(h)'
              ],axis = 1,inplace=True)



print('———————处理table6凝血————————')
table6.drop( ['术后条码','POD1条码','POD2条码','POD3条码','POD4条码','POD5条码','POD6条码'
            ,'POD7条码','POD14条码'],axis = 1,inplace=True)
#print(table6.columns.values)

print('——————— 处理table7术后输血情况 ————————')
table7.drop( ['病案号' ],axis = 1,inplace=True)#冗余的列

print('——————— 删除table8出院前术后转归 ————————')
table8.drop( ['手术结束时刻','拔管时刻','术后死亡时间d','术后自动离院时间d','死亡或放弃治疗原因'
             , '其它并发症','胆漏', '胸腔积液','胆漏或腹腔包裹性积胆','术后出血', '伤口感染' ,
              '肺炎', '胸腔积液需要引流穿刺', '胆漏需要ERCP','开腹止血','开腹清除脓肿','TBIL≥10 mg/Dl',
              'INR≥1.6', 'ALT或AST≥2000IU/mL' ,'初期移植肝无功能', '全部胸腔积液' ],axis = 1,inplace=True)#冗余的列
#print(table8.columns.values)
# print('——————— 删除table12胸腔积液冗余的列 ————————')
# table12.drop( ['胸腔积液','胸腔积液需要引流穿刺','全部胸腔积液'],axis = 1,inplace=True)

# print('——————— 处理预测标签————————')
# cols = ['序列号','年龄','身高','体重','BMI','O型','A型','B型','AB型','RH(阳1阴0)','乙肝携带']
# label1 = table1[cols]
#
#
#
#
table1.index=table1['序列号']
table2 = table2.set_index(['序列号'])#这种方法设置后'序列号'列会消失, 刚好防止了之后合并的多个表格都有'序列号'
table3 = table3.set_index(['序列号'])
table4 = table4.set_index(['序列号'])
table5 = table5.set_index(['序列号'])
table6 = table6.set_index(['序列号'])
table7 = table7.set_index(['序列号'])
table8 = table8.set_index(['序列号'])
print('———————合并表格————————')
# new=pd.concat(dfs2, axis=1, sort=False, join='left')
# new_concat = pd.concat([table1, table2, table3, table4, table5, table6, table7, table8], axis=1, sort=False)
new_concat = pd.concat([table1, table2, table3, table4, table5, table6, table7], axis=1, sort=False)
print("tabel7:\n",new_concat.columns.values)

# print('———————  合并1,2  ————————')
# new=pd.merge(table1,table2,on=['序列号'], how='left')
# print(len(table1.columns))
# print(len(table2.columns))
# print(len(new.columns))
# print(len(new))
# #new.to_csv('./merge_1.csv')

# print('———————  合并3  ————————')
# new=pd.merge(new,table3,on=['序列号'], how='left')
# print(len(table3.columns))
# print(len(new.columns))
# print(len(new))
# #new.to_csv('./merge_2.csv')

# print('———————  合并4  ————————')
# new=pd.merge(new,table4,on=['序列号'], how='left')
# print(len(table4.columns))
# print(len(new.columns))
# print(len(new))
# #new.to_csv('./merge_3.csv')

# print('———————  合并5  ————————')
# new=pd.merge(new,table5,on=['序列号'], how='left')
# print(len(table5.columns))
# print(len(new.columns))
# print(len(new))
# #new.to_csv('./merge_4.csv')

# print('———————  合并6  ————————')
# new=pd.merge(new,table6,on=['序列号'], how='left')
# print(len(table6.columns))
# print(len(new.columns))
# print(len(new))
# #new.to_csv('./merge_5.csv')

# print('———————  合并7  ————————')
# new=pd.merge(new,table7,on=['序列号'],how='left')
# print(len(table7.columns))
# print(len(new.columns))
# print(len(new))
# #new.to_csv('./merge_6.csv')

# print('———————  合并8  ————————')
# new=pd.merge(new,table8,on=['序列号'], how='left')
# print(len(table8.columns))
# print(len(new.columns))
# print(len(new))
# #new.to_csv('./merge_7.csv')

# # # print('———————  合并9  ————————')
# # # new=pd.merge(new,table9,on=['序列号','姓名'], how='left')
# # # print(len(table9.columns))
# # # print(len(new.columns))
# # # print(len(new))
# # #new.to_csv('./merge_8.csv')
# #
# # # print('———————  合并10  ————————')  ##该表格只有20多条记录，暂不考虑
# # # new=pd.merge(new,table10,on=['序列号','姓名','病案号','ID号'], how='left')
# # # print(len(table10.columns))
# # # print(len(new.columns))
# # # print(len(new))

# # print('———————  合并12胸腔积液  ————————')
# # new=pd.merge(new,table12,on=['序列号'], how='left')
# # print(len(table12.columns))
# # print(len(new.columns))
# # print(len(new))

# new.to_csv('../data/all_data.csv')
new_concat.to_csv('./data/features.csv', index=False)

#
#
#
#
#
#
#
print('——————— 查看数据结构 ————————')

# medical_data=pd.read_csv('./data/features.csv',index_col=0)
# medical_data=pd.read_csv('./data/labels.csv',index_col=0)
medical_data=pd.read_csv('./data/features.csv')
medical_label=pd.read_csv('./data/labels.csv')
#medical_data.set_index(["序列号"], inplace=True)
# print(medical_data.columns.values)
# print(medical_data)
# print(medical_data.info)  #获取数据集的简单描述
# print(medical_data['性别(男1女0)'].value_counts())
#print(medical_data.columns.values)
#print(medical_data.describe())
# plt.hist(medical_data['所有胸腔积液情况'],bins=30)
plt.show()
