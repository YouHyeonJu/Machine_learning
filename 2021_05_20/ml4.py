import numpy as np
from sklearn import linear_model
regr=linear_model.LinearRegression()
com_p={
    'name':['A','B','C','D','E','F','G'],
    'horse pwer':[130,250,190,300,210,220,170],
    'efficiency':[16.3,10.2,11.1,7.1,12.1,13.2,14.2]
}
print(com_p.keys())