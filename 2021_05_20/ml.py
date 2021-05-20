import re
from sklearn import linear_model
import matplotlib.pyplot as plt
import numpy as np
regr= linear_model.LinearRegression()


x=[[164],[179],[162],[170]]
y=[53,63,55,59]
regr.fit(x,y)

coef=regr.coef_
intercept=regr.intercept_
print("학습을 통하여 구해진 선형 회귀 직선의 방정식은\n")
print("y =",coef,"* X +",intercept)

input_data=[[180],[185]]
result= regr.predict(input_data)
print(result)

xx=np.array(x)
xx=xx.flatten()
print(xx)
print(np.corrcoef([xx,y]))

print("상관계수 구해볼까? 2")
import seaborn as sns
import pandas as pd
df=pd.DataFrame({
    'height':xx,
    'weight':y
})
print(df.corr())
sns.heatmap(df.corr(),annot=True,cmap="YlGnBu")


print("학습데이터 x와 y데이터를 이용한 산점도")
plt.scatter(x,y,color="green",marker="*")
y_pred=regr.predict(x)
plt.scatter(x,y_pred,color='red')
plt.plot(x,y_pred,color='yellow',linewidth=3)
plt.title("linearregression of y =[0.55221745] * X -35.68669527896999")
plt.show()