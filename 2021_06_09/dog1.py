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

dog_classes={0:'Dachshund',1:'Samoyed',2:'Maltese'}
dachshund_length=[77,78,85,83,73,77,73,80]
dachshund_height=[25,28,19,30,21,22,17,35]

samoyed_length=[75,77,86,86,79,83,83,88]
samoyed_height=[56,57,50,53,60,53,49,61]

maltese_length=[34,38,38,41,30,37,41,35]
maltese_height=[22,25,19,30,21,24,28,18]

print('dachshund=============')
dachshund=zip(dachshund_length,dachshund_height)
l=list(dachshund)
X1=[list(x) for x in l]
print(X1)
y1=[0]*len(X1)
print(y1)
print('samoyed==============')
samoyed=zip(samoyed_length,samoyed_height)
l=list(samoyed)
X2=[list(x) for x in l]
y2=[1]*len(X2)
print(y2)
print('말티즈==============')
maltese=zip(maltese_length,maltese_height)
l=list(maltese)
X3=[list(x) for x in l]
y3=[2]*len(X3)
print(y3)

dogs=X1+X2+X3
labels=y1+y2+y3
knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(dogs,labels)
new_data=[[45,34],[70,59],[49,30],[80,27]]
result=knn.predict(new_data)
import matplotlib
import matplotlib.font_manager as fm
fm.get_fontconfig_fonts()
font_location="c:\\windows\\fonts\\H2GTRM.TTF"
font_name=fm.FontProperties(fname=font_location).get_name()
matplotlib.rc('font',family=font_name)
x=[45,70,49,60]
y=[34,59,30,27]
data=['길이 45,높이 34','길이 70 , 높이 59','길이 49 , 높이 30','길이 60, 높이 27']
plt.scatter(dachshund_length,dachshund_height,c='red',label='Dachshund')
plt.scatter(samoyed_length,samoyed_height,c='blue',marker='^',label='samoyed')
plt.scatter(maltese_length,maltese_height,c='green',marker='s',label='maltese')
plt.scatter(x,y,c='magenta',label='new Data')
for i in range(4):
    plt.text(x[i],y[i],data[i],color='green')
plt.xlabel('Length')
plt.ylabel('Height')
plt.title('Dog size')
plt.legend(loc='upper right')
plt.show()