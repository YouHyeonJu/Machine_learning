import numpy as np
from sklearn import linear_model
import seaborn as sns
regr=linear_model.LinearRegression()
com_p={
    'name':['A','B','C','D','E','F','G'],
    'horse power':[130,250,190,300,210,220,170],
    'weight': [1900,2600,2200,2900,2400,2300,2100],
    'efficiency':[16.3,10.2,11.1,7.1,12.1,13.2,14.2]
}
print(com_p.keys())

a= np.array(com_p['horse power'])
print(a)
print(a.shape)
print("열백터")
print("first")
col_vec1=a.reshape(7,1)
print(col_vec1)
print(col_vec1.shape)
print()
print("second")
col_vec=a[:,np.newaxis]
print(col_vec.shape)

X=col_vec
y=com_p['efficiency']
regr.fit(X,y)
coef=regr.coef_
intercept=regr.intercept_
score=regr.score(X,y)

print("계수 :",coef)
print("절편 :",intercept)
print("예측 점수:",score)
result=regr.predict([[270]])
print("270 마력 자동차의 예상 연비 : {0:.2f}".format(result[0]),'km/l')
X=[]
for i in range(7):
    X.append([com_p['horse power'][i],com_p['weight'][i]])

y= com_p['efficiency']
regr.fit(X,y)
coef=regr.coef_
intercept=regr.intercept_
score=regr.score(X,y)
print("계수 : ",coef)
print("절편 :",intercept)

