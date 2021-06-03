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
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2, random_state=4)

knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(X,y)
classes={0:'setosa',1:'versicolor',2:'virginica'}
x_new=[[3,4,5,2],[5,4,2,2]]
y_predict=knn.predict(x_new)
print(classes[y_predict[0]])
print(classes[y_predict[1]])
iris_df2= pd.DataFrame(data=np.c_[iris['data'],iris['target']],
columns=iris['feature_names']+['target'])
print(iris_df2.head())

print("picture 2")
sns.pairplot(iris_df2,
        diag_kind='kde',
        hue="target",
        palette='colorblind')
plt.show()
y_pred_all= knn.predict(iris.data)
scores=metrics.accuracy_score(iris.target,y_pred_all)
print("n_neighbors 가 5일때 정확도 : {0:.3f}".format(scores))
print()

conf_mat=confusion_matrix(iris.target,y_pred_all)
print(conf_mat)
plt.matshow(conf_mat)
plt.show()

dog_classes={0:'Dachshund',1:'Samoyed',2:'Maltese'}
dachshund_length=[77,78,85,83,73,77,73,80]
dachshund_height=[25,28,19,30,21,22,17,35]

samoyed_length=[75,77,86,86,79,83,83,88]
samoyed_height=[56,57,50,53,60,53,49,61]

maltese_length=[34,38,38,41,30,37,41,35]
maltese_height=[22,25,19,30,21,24,28,18]

print('dachshund=============')
dachshund=zip(dachshund_length,dachshund_height)#
l=list(dachshund)
X1=[list(x) for x in l]#
print(X1)

y1=[0]*len(X1)#
print(y1)
print()

print('samoyed==============')
samoyed=zip(dachshund_length,dachshund_height)
l=list(samoyed)
X2=[list(x) for x in l]
print(X2)

y2=[1]* len(X2)
print(y2)
print('말티즈==============')
maltese=zip(maltese_length,maltese_height)
l=list(maltese)
X3=[list(x) for x in l]
print(X3)

y3=[2]* len(X3)
print(y3)

dogs=X1+X2+X3
labels=y1+y2+y3

print("neightbor의 갯수===============>",5)
knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(dogs,labels)
new_data=[[45,34],[70,59],[49,30],[80,27]]
result=knn.predict(new_data)#
print("길이 45, 높이 34:{}".format(dog_classes[result[0]]))
print("길이 70, 높이 59:{}".format(dog_classes[result[1]]))
print("길이 49, 높이 30:{}".format(dog_classes[result[2]]))
print("길이 80, 높이 27:{}".format(dog_classes[result[3]]))
import matplotlib
import matplotlib.font_manager as fm
fm.get_fontconfig_fonts()
font_location="c:\\windows\\fonts\\H2GTRM.TTF"
font_name=fm.FontProperties(fname=font_location).get_name()
matplotlib.rc('font',family=font_name)
x=[45,70,49,60]
y=[34,59,30,27]
data=['길이 45 ,높이 34', '길이 70 ,높이 59' ,'길이 49 ,높이 30 ','길이 60 ,높이 27']

plt.scatter(dachshund_length,dachshund_height,c='red',label='Dachshund')
plt.scatter(samoyed_length,samoyed_height,c='blue',marker='^',label='Samoyed')
plt.scatter(maltese_length,maltese_height,c='green',marker='s',label='Maltese')

plt.scatter(x,y,c='magenta',label='new Data')
for i in range(4):
    plt.text(x[i],y[i],data[i],color='green')#x[i],y[i],data[i]
plt.xlabel('Length')
plt.ylabel('Hight')
plt.title("Dog size")
plt.legend(loc='upper left')
plt.show()

from sklearn import linear_model
regr=linear_model.LinearRegression()
X=[[174],[152],[138],[128],[186]]
y=[71,55,46,38,88]
regr.fit(X,y)
coef=regr.coef_
intercept=regr.intercept_

score= regr.score(X,y)#
print("선형회귀방정식의 적합도",score)
print()
plt.scatter(X,y,color='green',marker='*')
y_pred=regr.predict(X)#
plt.plot(X,y_pred,color='yellow',linewidth=3)

plt.title('LinearRegression')
plt.show()


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

regr.fit(X,y)#
score=regr.score(X,y)#
result=regr.predict([[270],[300]])###
print("270 마력 자동차의 예상 연비 : {0:.2f}".format(result[0]),'km/l')
print('300 마력 자동차의 예상 연비 : {0:.2f}'.format(result[1]),'km/l')


df=pd.DataFrame({
    'name':['A','B','C','D','E','F','G','H'],
    'horse power':[130,250,190,300,210,220,170,200],
    'weight':[1900,2600,2200,2900,2400,2300,2100,2300],
    'efficiency':[16.3,10.2,11.1,7.1,12.1,13.2,14.2,15.2]
})
print(df)
X=df[['horse power','weight']]#
y=df['efficiency']
regr.fit(X,y)
score=regr.score(X,y)
print("예측모델의 적합도 :",score)
result=regr.predict([[270,2500]])
print("270마력 2500kg 자동차의 예상연비 : {0:.2f}".format(result[0]),'km/l')
print()

sns.pairplot(df[['horse power','weight','efficiency']])
plt.show()
print(df.corr())
sns.heatmap(df.corr(),annot=True,cmap="YlGnBu",linewidths=2)##df.corr()
plt.show()

from sklearn.model_selection import train_test_split
df=pd.DataFrame({
    'name':['A','B','C','D','E','F','G','H','I','K'],
    'horse power':[130,250,190,300,210,220,170,200,300,290],
    'weight':[1900,2600,2200,2900,2400,2300,2100,2300,2800,2700],
    'efficiency':[16.3,10.2,11.1,7.1,12.1,13.2,14.2,15.2,8.1,9.0]
})
X=df[['horse power','weight']]#
y=df['efficiency']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)

print(df)

regr.fit(X_train,y_train)
score=regr.score(X_train,y_train)
print("예측모델의 적합도 :",score)
result=regr.predict([[270,2500]])
print("270마력 2500kg 자동차의 예상연비 : {0:.2f}".format(result[0]),'km/l')
print()

sns.pairplot(df[['horse power','weight','efficiency']])
plt.show()
print(df.corr())
sns.heatmap(df.corr(),annot=True,cmap="YlGnBu",linewidths=2)##df.corr()
plt.show()