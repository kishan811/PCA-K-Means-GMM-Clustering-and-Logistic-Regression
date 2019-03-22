#!/usr/bin/env python
# coding: utf-8

# ## Part 2: KNN on Admission Dataset :-

# In[9]:


import numpy as np  
import matplotlib.pyplot as plt  
import pandas as pd
import csv
import random
import math
import operator
from numpy import linalg as LA
import sys
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from sklearn import linear_model


# In[10]:


df = pd.read_csv('AdmissionDataset/data.csv') 


# In[11]:


df.head()


# In[12]:


# logisticRegr.fit(train, train_y)
# logisticRegr.predict(test[0].reshape(1,-1))


# In[13]:


df.head()
X = df.iloc[:, :-1].values  
y = df.iloc[:, -1].values 
y=np.where(y>=0.5,1,0)


# In[14]:


from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2) 


# In[15]:


def euclidDistance(val1, val2, entirelength):
    distance = 0
    for x in range(entirelength):
        distance += pow((val1[x] - val2[x]), 2)
    return math.sqrt(distance)
def Neighbor_points(train_d, test_d, k):
    distances = []
    length = len(test_d)-1
    for x in range(len(train_d)):
        dist = euclidDistance(test_d, train_d[x], length)
        distances.append((train_d[x], dist))
    distances.sort(key=lambda x: x[1])
#     print(distances)
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
#         print(neighbors)
    return neighbors
def resultss(neighbors):
    predlabel = {}
    for x in range(len(neighbors)):
        res = neighbors[x][-1]
        if res not in predlabel:
            predlabel[res] = 1
        predlabel[res] += 1
    ans = sorted(predlabel.items(), reverse=True)
    return ans[0][0]
def accu_calc(test, predlabel):
    correct = 0
    for x in range(len(test)):
        if test[x][-1] == predlabel[x]:
            correct += 1
    return (correct/float(len(test))) * 100.0


# In[16]:


train,test = train_test_split(df, test_size=0.2) 
pre_values=[]
k = 5
train['Chance of Admit ']=np.where(train['Chance of Admit ']>=0.5,1,0)
test['Chance of Admit ']=np.where(test['Chance of Admit ']>=0.5,1,0)
test1=test
train1=train
test=test.values
train=train.values

for x in range(len(test)):
    neighbors = Neighbor_points(train, test[x], k)
    result = resultss(neighbors)
    pre_values.append(result)
accuracy = accu_calc(test, pre_values)
# print(pre_values)
print ('The Accuracy is: ', accuracy)
# test


# In[ ]:




