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
from sklearn import linear_model
regr=linear_model.LinearRegression()
X=[[174],[152],[138],[128],[186]]
y=[71,55,46,38,88]

regr.fit(X,y)
coef=regr.coef_
intercept=regr.intercept_
print(coef)
print(intercept)
score=regr.score(X,y)
print(score)
plt.scatter(X,y,color='green',marker='*')
y_pred=regr.predict(X)
print(y_pred)
plt.plot(X,y_pred,color='yellow',linewidth=3)
plt.title('LinearRegression')
plt.show()