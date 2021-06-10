from functools import cmp_to_key
from scipy.sparse import data
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import seaborn as sns
from sklearn.metrics import confusion_matrix

iris=load_iris()
X=iris.data
y=iris.target
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.4,random_state=4)
knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(X,y)
score=knn.score(X,y)
classes={0:'setosa',1:'versicolor',2:'virginica'}
x_new=[[3,4,5,2],[5,4,2,2]]
y_predict=knn.predict(x_new)
print(classes[y_predict[0]])
print(classes[y_predict[1]])
iris_df2=pd.DataFrame(data=np.c_[iris['data'],iris['target']],
columns=iris['feature_names']+['target'])
print(iris_df2.head())

print("picture 2")
sns.pairplot(iris_df2,
    diag_kind='kde',
    hue='target',
    palette='colorblind')
plt.show()
y_pred_all=knn.predict(iris.data)
score=metrics.accuracy_score(iris.target,y_pred_all)
print("n_neighbors 가 5일때 정확도 : {0:.3f}".format(score))
print()
conf_mat=confusion_matrix(iris.target,y_pred_all)
print(conf_mat)
plt.matshow(conf_mat)
plt.show()