# from sklearn.datasets import fetch_mldata
# mnist = fetch_mldata('MNIST original')
from sklearn.datasets import fetch_openml
mnist = fetch_openml("mnist_784")

X, y = mnist["data"], mnist["target"]
X.shape
(70000, 784)
y.shape
(70000,)

X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

y_train_5 = (y_train == 5) # True for all 5s, False for all other digits.
y_test_5 = (y_test == 5)

from sklearn.linear_model import SGDClassifier
sgd_clf = SGDClassifier(random_state=42)
# sgd_clf.fit(X_train, y_train_5)
sgd_clf.fit(X_train, y_train)
# StratifiedKFold类实现了分层采样（详见第二章的解释），生成的折（fold）包含了各类相应比例的样例。
# 在每一次迭代，上述代码生成分类器的一个克隆版本，在训练折（training folds）的克隆版本上进行训练，在测试折（test folds）上进行预测。然后它计算出被正确预测的数目和输出正确预测的比例。
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
skfolds = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
# for train_index, test_index in skfolds.split(X_train, y_train_5):
for train_index, test_index in skfolds.split(X_train, y_train):
    clone_clf = clone(sgd_clf)
    # X_train_folds = X_train[train_index]
    # y_train_folds = (y_train_5[train_index])
    X_train_folds = X_train.iloc[train_index]
    y_train_folds = (y_train.iloc[train_index])


    # X_test_fold = X_train[test_index]
    # y_test_fold = (y_train_5[test_index])
    X_test_fold = X_train.iloc[test_index]
    y_test_fold = (y_train.iloc[test_index])

    clone_clf.fit(X_train_folds, y_train_folds)
    y_pred = clone_clf.predict(X_test_fold)
    n_correct = sum(y_pred == y_test_fold)
    print(n_correct / len(y_pred)) # prints 0.9502, 0.96565 and 0.96495
# (对比)使用cross_val_score()函数来评估SGDClassifier模型，同时使用 K 折交叉验证，此处让k=3
from sklearn.model_selection import cross_val_score
# cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring="accuracy")
cross_val_score(sgd_clf, X_train, y_train, cv=3, scoring="accuracy")
# array([ 0.9502 , 0.96565, 0.96495]
