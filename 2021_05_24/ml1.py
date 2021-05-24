from sklearn import linear_model
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
regr= linear_model.LinearRegression()
X=[[164],[179],[162],[170],[175]]
y=[53,63,55,59,62]
regr.fit(X,y)
coef=regr.coef_
intercept=regr.intercept_
print("학습을 통하여 구해진 선형 회귀 직선의 방정식은 \n")
print("y=",coef,"*X+",intercept)

score=regr.score(X,y)
print("The score of this line for the data:",score)

input_data=[[180],[185]]
result=regr.predict(input_data)
print(result)


xx=np.array(X)
xx=xx.flatten()
print(xx)
print(np.corrcoef([xx,y]))
df= pd.DataFrame({
    'height':xx,
    'weight':y
})
print(df.corr())
sns.heatmap(df.corr(),annot=True,cmap="YlGnBu")

plt.show()
