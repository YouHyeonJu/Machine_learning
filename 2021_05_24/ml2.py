from sklearn import linear_model
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
regr= linear_model.LinearRegression()
X=[[164,1],[167,1],[165,0],[170,0],[179,0],[159,0],[166,1]]
y=[43,48,47,66,67,50,52,44]
regr.fit(X,y)
coef=regr.coef_
intercept=regr.intercept_
print("계수 : ",coef)
print("절편 :",intercept)
print("학습을 통하여 구해진 다차원 선형 회귀 방정식")
print("y=",coef,"*X+",intercept)
print("적합도:",regr.score(X,y))
print()