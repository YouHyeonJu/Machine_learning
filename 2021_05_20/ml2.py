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

print("적합도",regr.score(x,y))
input_data=[[166,1],[166,0]]
print("추정몸무게 :",regr.predict(input_data))