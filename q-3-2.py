#!/usr/bin/env python
# coding: utf-8

# ## One vs One Logistic

# In[69]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score,confusion_matrix,classification_report,accuracy_score
df=pd.read_csv("wine-quality/data.csv",delimiter=';')
lrr = 0.01
itr = 1000
th = 0.5
df.head()


# In[70]:


X = df.drop(['quality'],axis=1)
Y = df['quality']
print(df.shape)
X.shape


# In[71]:


X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = 0.2)


# In[72]:


ones = np.ones([X_train.shape[0],1])
X_train = np.concatenate((ones,X_train),axis=1)
X_train.shape


# In[73]:


theta = np.zeros([15,12])
theta.shape


# In[74]:


def sigmoid(z):
    return 1 / (1 + np.exp(-z))
def grad_des(X_train,Y_train,thh,lrr,itr): 
    for i in range(itr):
        h = sigmoid(np.dot(X_train,thh.T))
        thh = thh - (lrr/len(X_train)) * np.sum(X_train * (h - Y_train), axis=0)  
    return thh


# In[75]:


x=0
for i in range(4,9):
    for j in range(i+1,10):
        thh = np.zeros([1,12])
        temp1 = []
        temp2 = []
        W = np.array(Y_train)
        for k in range(len(W)):
            if W[k] == j or W[k]==i:
                temp1.append(X_train[k])
                if W[k] == i:
                    temp2.append(1)
                else:
                    temp2.append(0)
        
        temp1 = np.array(temp1)
        temp2 = np.array(temp2)
        temp2 = temp2.reshape((len(temp2),1))
        theta[x]=grad_des(temp1,temp2,thh,lrr,itr)
        x = x + 1


# In[76]:


y_pred = []
for index,rows in X_test.iterrows():
    rows = list(rows)
    counts = {}
    label = 0
    for i in range(4,10):
        counts[i]=0
    h = 0
    c = 0
    for a in range(4,9):
        for b in range(a+1,10):
            y = 0
            for i in range(len(rows)):
                y = y + rows[i]*theta[c][i+1]
            y = y + theta[c][0]
            y = sigmoid(y)
            c = c + 1
            if y >= th:
                counts[a]=counts[a]+1
            else:
                counts[b]=counts[b]+1
    for i in range(4,10):
        if(counts[i]>=h):
            h=counts[i]
            label=i
    y_pred.append(label)
# y_pred
# counts


# In[78]:


# print((y_pred == Y_test).mean()*100)
print(confusion_matrix(Y_test,y_pred))
# print(classification_report(Y_test,y_pred))
print("Accuracy: ",accuracy_score(Y_test, y_pred)*100)


# In[ ]:




