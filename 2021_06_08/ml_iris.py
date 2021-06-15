from functools import cmp_to_key
from operator import irshift
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
knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(X,y)

classes={0:'setosa',1:'versicolor',2:'virginica'}
x_new=[[3,4,5,2],[5,4,2,2]]
y_pred=knn.predict(x_new)

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
