#!/usr/bin/env python
# coding: utf-8

# ## Part 1: Logistic Regression

# In[660]:


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


# In[661]:


df = pd.read_csv('AdmissionDataset/data.csv') 


# In[662]:


df.head()


# In[ ]:





# In[663]:


from sklearn.model_selection import train_test_split  
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2) 
train,test = train_test_split(df, test_size=0.2)
train_y = train['Chance of Admit ']
train = train.drop('Serial No.',axis=1)
train = train.drop('Chance of Admit ',axis=1)
test_y = test['Chance of Admit ']
test = test.drop('Serial No.',axis=1)
test = test.drop('Chance of Admit ',axis=1)
train_y=np.where(train_y>=0.5,1,0)
test_y=np.where(test_y>=0.5,1,0)
# train_y.values


# In[664]:


X=train.values
X1=test.values
# print(X)
z1= np.ones((len(test),1))
X1= np.append(z1,X1, axis=1)
z = np.ones((len(train),1))
X=np.append(z,X, axis=1)
Y=train_y
X.shape


# In[665]:


def sigmoid(z):
    return 1 / (1 + np.exp(-z))
# print(beta)

def loss(h, y):
    return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()


# In[666]:


y=train_y
beta = np.zeros(X.shape[1]) 
print((X.shape[1]))


# In[667]:


# beta, num_iter = grad_desc(X, y, beta)
lr=0.01


# In[668]:


for i in range(30000):
    z = np.dot(X, beta)
#     print(z.shape)
    h = sigmoid(z)
    gradient = np.dot(X.T, (h - y)) / y.size
    beta -= lr * gradient
beta


# In[669]:


z = np.dot(X1, beta)
h = sigmoid(z)
ans=sigmoid(np.dot(X1, beta))
ans=np.where(ans>=0.5,1,0)
# ans


# In[670]:


# ans


# In[671]:


c=0
for i in range(len(X1)):
    if (test_y[i]==ans[i]):
        c+=1
accu=c/len(X1)
print("Accuracy using Logistic Regression: ",accu)


# In[672]:


reg = linear_model.LogisticRegression(solver='lbfgs') 
# train the model using the training sets 
reg.fit(train,train_y)
pred2=reg.predict(test)
print("Acc using SKlearn: ", accuracy_score(test_y,pred2))


# In[673]:


# logisticRegr.fit(train, train_y)
# logisticRegr.predict(test[0].reshape(1,-1))


# In[674]:


# df.head()
# X = df.iloc[:, :-1].values  
# y = df.iloc[:, -1].values 
# y=np.where(y>=0.5,1,0)


# In[675]:


# from sklearn.model_selection import train_test_split  
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2) 


# In[676]:


# def euclidDistance(val1, val2, entirelength):
#     distance = 0
#     for x in range(entirelength):
#         distance += pow((val1[x] - val2[x]), 2)
#     return math.sqrt(distance)
# def Neighbor_points(train_d, test_d, k):
#     distances = []
#     length = len(test_d)-1
#     for x in range(len(train_d)):
#         dist = euclidDistance(test_d, train_d[x], length)
#         distances.append((train_d[x], dist))
#     distances.sort(key=lambda x: x[1])
# #     print(distances)
#     neighbors = []
#     for x in range(k):
#         neighbors.append(distances[x][0])
# #         print(neighbors)
#     return neighbors
# def resultss(neighbors):
#     predlabel = {}
#     for x in range(len(neighbors)):
#         res = neighbors[x][-1]
#         if res not in predlabel:
#             predlabel[res] = 1
#         predlabel[res] += 1
#     ans = sorted(predlabel.items(), reverse=True)
#     return ans[0][0]
# def accu_calc(test, predlabel):
#     correct = 0
#     for x in range(len(test)):
#         if test[x][-1] == predlabel[x]:
#             correct += 1
#     return (correct/float(len(test))) * 100.0


# In[677]:


# train,test = train_test_split(df, test_size=0.2) 
# pre_values=[]
# k = 5
# train['Chance of Admit ']=np.where(train['Chance of Admit ']>=0.5,1,0)
# test['Chance of Admit ']=np.where(test['Chance of Admit ']>=0.5,1,0)
# test1=test
# train1=train
# test=test.values
# train=train.values

# for x in range(len(test)):
#     neighbors = Neighbor_points(train, test[x], k)
#     result = resultss(neighbors)
#     pre_values.append(result)
# accuracy = accu_calc(test, pre_values)
# # print(pre_values)
# print ('The Accuracy is: ', accuracy)
# # test


# In[ ]:




