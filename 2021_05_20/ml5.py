import numpy as np
from sklearn import linear_model
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
regr=linear_model.LinearRegression()
com_p=pd.DataFrame({
    'name':['A','B','C','D','E','F','G'],
    'horse power':[130,250,190,300,210,220,170],
    'weight': [1900,2600,2200,2900,2400,2300,2100],
    'efficiency':[16.3,10.2,11.1,7.1,12.1,13.2,14.2]
})
sns.pairplot(com_p[['horse power','weight','efficiency']])
plt.show()
print(com_p.corr())
sns.heatmap(com_p.corr(),annot=True,cmap="YlGnBu")
plt.show()