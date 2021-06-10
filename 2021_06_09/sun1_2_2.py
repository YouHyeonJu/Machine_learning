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
com_p={
    'name':['A','B','C','D','E','F','G','H'],
    'horse power':[130,250,190,300,210,220,170,200],
    'weight':[1900,2600,2200,2900,2400,2300,2100,2300],
    'efficiency':[16.3,10.2,11.1,7.1,12.1,13.2,14.2,15.2]
}
X=[]
for i in range(8):
    X.append([com_p['horse power'][i]])
print(X)
y=com_p['efficiency']
print(y)
regr.fit(X,y)
score=regr.score(X,y)
print(score)
result=regr.predict([[270],[300]])
print(result)
print("270 마력 자동차의 예상 연비 : {0:.2f}".format(result[0]),'km/l')
print('300 마력 자동차의 예상 연비 : {0:.2f}'.format(result[1]),'km/l')

df=pd.DataFrame({
    'name':['A','B','C','D','E','F','G','H'],
    'horse power':[130,250,190,300,210,220,170,200],
    'weight':[1900,2600,2200,2900,2400,2300,2100,2300],
    'efficiency':[16.3,10.2,11.1,7.1,12.1,13.2,14.2,15.2]
})
print(df)
X=df[['horse power','weight']]
y=df['efficiency']
regr.fit(X,y)
score=regr.score(X,y)
result=regr.predict([[270,2500]])
sns.pairplot(df[['horse power','weight','efficiency']])
plt.show()
sns.heatmap(df.corr(),annot=True,cmap='YlGnBu',linewidths=3)
plt.show()

