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
df=pd.DataFrame({
    'name':['A','B','C','D','E','F','G','H','I','K'],
    'horse power':[130,250,190,300,210,220,170,200,300,290],
    'weight':[1900,2600,2200,2900,2400,2300,2100,2300,2800,2700],
    'efficiency':[16.3,10.2,11.1,7.1,12.1,13.2,14.2,15.2,8.1,9.0]
})
X=df[['horse power','weight']]
y=df['efficiency']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
print(df)
regr=linear_model.LinearRegression()
regr.fit(X_train,y_train)
score=regr.score(X_train,y_train)
print(score)
result=regr.predict([[270,2500]])
print("270마력 2500kg 자동차의 예상연비 : {0:.2f}".format(result[0]),'km/l')
print()
sns.pairplot(df[['horse power','weight','efficiency']])
plt.show()
print(df.corr())
sns.heatmap(df.corr(),annot=True,cmap='YlGnBu',linewidths=2)
plt.show()