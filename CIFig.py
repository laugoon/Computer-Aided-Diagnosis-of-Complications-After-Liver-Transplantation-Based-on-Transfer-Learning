import numpy as np
import matplotlib.pyplot as plt
label_list=['Pleural effusion','High INR','High ALT or AST','High TBIL','Pneumonia','Postoperative bleeding','Postoperative infection','Biliary complications','PNF']


x=label_list
y=[86.12,79.05, 97.18,94.83,89.64,99.29,98.82,100.00,99.77]
dy=[[5.88,6.82, 2.82,4.00,5.18,1.18,1.41,0.00,0.47],[5.18,6.59, 2.12,3.06,4.71,0.71,0.94,0.00,0.23]]

plt.errorbar(x,y,yerr=dy,fmt='o',ecolor='r',color='b',elinewidth=2,capsize=4)
# 图片设置
# # 图片大小
# plt.figure(figsize=(0.9 * max_display + 1, 0.8* max_display + 1))
# # 添加数值标签
# for i, v in enumerate(y_values):
#     plt.text(i, v+0.02*y_max , str("%.3f" %v), ha='center')
#调整 图表 的上下左右
plt.subplots_adjust(bottom=0.45)
# 添加y轴标签
plt.ylabel('Accuracy(%)')
plt.ylim((0,105))
plt.xticks(rotation=90)
# plt.show()
plt.savefig("./fig/CI/ACC_CI.pdf")
