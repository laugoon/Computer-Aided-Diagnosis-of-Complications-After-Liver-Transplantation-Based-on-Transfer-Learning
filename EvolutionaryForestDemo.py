import random

import numpy as np

from lightgbm import LGBMRegressor
from sklearn.datasets import load_diabetes
from sklearn.ensemble import ExtraTreesRegressor, AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from evolutionary_forest.utils import get_feature_importance, plot_feature_importance, feature_append
from evolutionary_forest.forest import cross_val_score, EvolutionaryForestRegressor

import matplotlib.pyplot as plt
import seaborn as sns
# 模型训练
random.seed(0)
np.random.seed(0)

X, y = load_diabetes(return_X_y=True)
print('y:',y)
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
r = RandomForestRegressor()
r.fit(x_train, y_train)
print(r2_score(y_test, r.predict(x_test)))
r = EvolutionaryForestRegressor(max_height=8, normalize=True, select='AutomaticLexicase',
                                mutation_scheme='weight-plus-cross-global',
                                gene_num=10, boost_size=100, n_gen=100, base_learner='DT',
                                verbose=True)
print('X:',x_train.shape,x_test.shape)
print('y:',y_train.shape,y_test.shape )
r.fit(x_train, y_train)
print(r2_score(y_test, r.predict(x_test)))
# 进行排序(显示前15个)
feature_importance_dict = get_feature_importance(r)
# plot_feature_importance(feature_importance_dict)

# 利用构造好的新特征(最重要20个)，改进现有模型的性能
code_importance_dict = get_feature_importance(r, simple_version=False)
new_X = feature_append(r, X, list(code_importance_dict.keys())[:20], only_new_features=True)
new_train = feature_append(r, x_train, list(code_importance_dict.keys())[:20], only_new_features=True)
new_test = feature_append(r, x_test, list(code_importance_dict.keys())[:20], only_new_features=True)
new_r = RandomForestRegressor()
new_r.fit(new_train, y_train)
print(r2_score(y_test, new_r.predict(new_test)))
# 新特征应用于其他机器学习
regressor_list = ['RF', 'ET', 'AdaBoost', 'GBDT', 'DART', 'XGBoost', 'LightGBM', 'CatBoost']

# scores_base = []
# scores = []

# for regr in regressor_list:
#     regressor = {
#         'RF': RandomForestRegressor(n_jobs=1, n_estimators=100),
#         'ET': ExtraTreesRegressor(n_estimators=100),
#         'AdaBoost': AdaBoostRegressor(n_estimators=100),
#         'GBDT': GradientBoostingRegressor(n_estimators=100),
#         'DART': LGBMRegressor(n_jobs=1, n_estimators=100, boosting_type='dart'),
#         'XGBoost': XGBRegressor(n_jobs=1, n_estimators=100),
#         'LightGBM': LGBMRegressor(n_jobs=1, n_estimators=100),
#         'CatBoost': CatBoostRegressor(n_estimators=100, thread_count=1,
#                                       verbose=False, allow_writing_files=False),
#     }[regr]
#     score = cross_val_score(regressor, X, y)
#     print(regr, score, np.mean(score))
#     scores_base.append(np.mean(score))
#     score = cross_val_score(regressor, new_X, y)
#     print(regr, score, np.mean(score))
#     scores.append(np.mean(score))
# scores_base = np.array(scores_base)
# scores = np.array(scores)

# import matplotlib.pyplot as plt
# import seaborn as sns

# sns.set(style="white", font_scale=1.5)
# width = 0.4
# fig, ax = plt.subplots(figsize=(10, 6))
# ax.bar(regressor_list, scores_base, width, label='Original Features')
# difference = scores - scores_base
# print(np.where(difference > 0, 'g', 'y'))
# ax.bar(regressor_list, difference, width, bottom=scores_base,
#        label='Constructed Features',
#        color=np.where(difference > 0, 'r', 'y'))
# ax.set_ylabel('Score ($R^2$)')
# ax.set_title('Effect of Feature Construction')
# ax.legend()
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.show()