from seaborn.axisgrid import pairplot
from sklearn import linear_model
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
regr= linear_model.LinearRegression()
df=pd.DataFrame ({
    "name":["A","B","C","D","E","F","G"],
    "horse power":[130,250,190,300,210,220,170],
    "weight":[1900,2600,2200,2900,2400,2300,2100],
    'enfficiency':[16.3,10.2,11.1,7.1,12.1,13.2,14.2]
})
print(df)
print(df.keys())

X=df[['horse power','weight']]
print(X)
y=df['enfficiency']
print(y)
regr.fit(X,y)
coef=regr.coef_
intercept=regr.intercept_
print("계수 : ",coef)
print("절편 :",intercept)
print()
score=regr.score(X,y)
print("예측모델의 적합도 점수 :",score)
print("예측하기")
sns.pairplot(df[['horse power','weight','enfficiency']])
plt.show()
print(df.corr())
sns.heatmap(df.corr(),annot=True,cmap="YlGnBu")
plt.show()