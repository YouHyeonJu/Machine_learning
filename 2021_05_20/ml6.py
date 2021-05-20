import re
import numpy as np
from sklearn import linear_model
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
regr=linear_model.LinearRegression()

diabetes=datasets.load_diabetes()
print("diabetes의 키 \n{}".format(diabetes.keys()))

print('입력데이터')
print(diabetes.data)
print('shape of diabetes.data: ',diabetes.data.shape)

print()
print("입력데이터의 특성들")
print("당뇨 수치에 영향을 줄 수 있는 각종 검사값")
print(diabetes.feature_names)
print('diabetes 의 feature names\n{}'.format(diabetes.feature_names))
print(
)
print("학습시 결과값으로 사용되는 target 데이터")
print(diabetes.target)
print("target data y:",diabetes.target.shape)
print('diabetes의 DESCR\n{}'.format(diabetes['DESCR']))

print()
print('10가지의 특성 중에서 체질량 지수에 해당되는 세번째 항목만 추출')
print("체질량지수 bmi")
bmi=diabetes.data[:,-2]
print(bmi)
print(bmi.shape)
print()
tar=diabetes.target
print(tar.shape)
print()
print('상관계수를 구해볼까')
result=np.corrcoef([bmi,tar])
print(result)
print()
df=pd.DataFrame({
    'bmi':bmi,
    'target':tar
})
print(df.corr())
sns.heatmap(df.corr(),annot=True,cmap="YlGnBu")
plt.show()
print()
print('0')
x1=diabetes.data[:,2]
X=x1.reshape(442,1)
print(X.shape)
print('======================================')
print("1")
xx=diabetes.data[:,2]
X=xx[:,np.newaxis]
print(X.shape)
print("==================================")
print("2")
X=diabetes.data[:,np.newaxis,2]
print(X.reshape)
X_train,X_test,y_train,y_test=train_test_split(diabetes.data[:,np.newaxis,2],diabetes.target,test_size=0.2)
print('X_train의 크기: {}'.format(X_train.shape))
print('X_test의 크기: {}'.format(X_test.shape))
print('y_train의 크기: {}'.format(y_train.shape))
print('y_test의 크기: {}'.format(y_test.shape))
regr=linear_model.LinearRegression()
regr.fit(X_train,y_train)
coef=regr.coef_
intercept=regr.intercept_
print("당뇨수치 =",coef,"* 체질량 지수 +",intercept)
print()

print("훈련 데이터만 가지고 적합도 계산")
print("계수와 기울기 값들이 bmi가 당뇨수치를 예측하는데 얼마나 적합하는가")
print()
score = regr.score(X_train,y_train)
print("The score of this line for rhe train data:", score)
print()
print("테스트 데이터만 가지고 적합도 계산")
score =regr.score(X_test,y_test)
print("The score of this line for the test data:",score)

plt.scatter(X_train,y_train,color='green',marker='*')
y_pred=regr.predict(X_train)
plt.plot(X_train,y_pred,color='yellow',linewidth=3)
plt.title("linearregression of bmi and diabetes values")
plt.show()



y_pred=regr.predict(X_test)
plt.scatter(y_pred,y_test,color='red')
x=np.linspace(0,330,100)
plt.plot(x,x,linewidth=3,color='blue')
plt.show()

print("예측값과 실제값사이에 오차가 있음")
print("Mean squared error:",mean_squared_error(y_test,y_pred))