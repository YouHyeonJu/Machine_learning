from seaborn.axisgrid import pairplot
from sklearn import linear_model
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
regr= linear_model.LinearRegression()
iris=load_iris()
print("iris의 keys \n{}".format(iris.keys()))
print()

print(type(iris.data))
print("iris의 data 의 크기 \n{}".format(iris['data'].shape))
print(iris.data[:3,:])
print()
print("feature_names")
print(iris.feature_names)
print()
print("iris의 target names")
print("iris의 target names \n{}".format(iris.target_names))
print('0=setosa,1=versicolor,2=virginica')
print()
print('iris의 target의 크기\n{}'.format(iris.target.shape))
print(type(iris.target))
print(iris.target[:5])
print()
print("iris의 DESCR \n{}".format(iris['DESCR']))

iris_df=pd.DataFrame(iris.data,columns=iris.feature_names)
print(iris_df.head())
iris_df2=pd.DataFrame(data=np.c_[iris['data'],iris['target']], columns=iris['feature_names']+['target'])
print(iris_df2.head())

X_train,X_test,y_train,y_test=train_test_split(iris['data'],iris['target'])
print("X_train의 크기 : {}".format(X_train.shape))
print("y_train의 크기 : {}".format(y_train.shape))
print("X_test의 크기 : {}".format(X_test.shape))
print("y_test의 크기 : {}".format(y_test.shape))

print("picture 2")
sns.pairplot(iris_df2,
        diag_kind='kde',
        hue="target",
        palette="colorblind")
plt.show()

from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

# num_neigh=1
# knn=KNeighborsClassifier(n_neighbors=num_neigh)
# knn.fit(X_train,y_train)
# print("테스트 데이터를 이용하여 예측")
# y_pred=knn.predict(X_test)
# scores=metrics.accuracy_score(y_test,y_pred)

print()
for i in range(1,11):
    num_neigh=i
    knn=KNeighborsClassifier(n_neighbors=num_neigh)
    knn.fit(X_train,y_train)
    y_pred=knn.predict(X_test)
    scores=metrics.accuracy_score(y_test,y_pred)
    print("n_neighbors가 {0:d}일때 정확도: {1:.3f}".format(num_neigh,scores))