from sklearn import linear_model
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
regr= linear_model.LinearRegression()
com_p= {
    "name":["A","B","C","D","E","F","G"],
    "horse power":[130,250,190,300,210,220,170],
    "weight":[1900,2600,2200,2900,2400,2300,2100],
    'enfficiency':[16.3,10.2,11.1,7.1,12.1,13.2,14.2]
}
print(com_p)
print(com_p.keys())

X=[]
for i in range(7):
    X.append([com_p['horse power'][i]])
y= com_p['efficiency']
regr.fit(X,y)
coef=regr.coef_
intercept=regr.intercept_
print("계수 : ",coef)
print("절편 :",intercept)
result=regr.predict([[270]])
print("270 마력 자동차의 예상 연비:{0:.2f}".format(result[0]),'km/l')
print()
X=[]
for i in range(7):
    X.append([com_p['horse power'][i]],com_p['weight'][i])
print(X)
y=com_p['efficiency']
regr.fit(X,y)
coef=regr.coef_
intercept=regr.intercept_
print()
result=regr.predict([[270,2500]])
print("270 마력 2500kg 자동차의 예상 연비:{0:.2f}".format(result[0]),'km/l')
print()