#!/usr/bin/env python
# coding: utf-8

# ## One vs All Logistic

# In[528]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score,confusion_matrix,classification_report,accuracy_score
from sklearn.preprocessing import StandardScaler


# In[529]:


df = pd.read_csv('wine-quality/data.csv',delimiter=';')
df.head()


# In[530]:


lrr=0.01
itr=1000
X = df.drop(['quality'],axis=1)
Y = df['quality']


# In[531]:


X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = 0.2)
# Y_train = df['quality']
# Y_train
A=X_train
B=X_test
C=Y_train
D=Y_test
def sklearn():
    lr = LogisticRegression(solver='lbfgs',multi_class='multinomial',max_iter=itr)
    lr.fit(A, C) 
    y_pred = lr.predict(B)
    score = lr.score(B,D)
    print("Score: ",score)
    print((D == y_pred).mean())
    print(confusion_matrix(D,y_pred))
    print(classification_report(D,y_pred))
    print("Accuracy using SK learn: ",accuracy_score(D, y_pred)*100)


# In[532]:


X_train = pd.concat([X_train,Y_train],axis=1)
ones = np.ones([X_train.shape[0],1])
Y_train = X_train.iloc[:,11:12].values
X_train = X_train.iloc[:,0:11]
X_train = np.concatenate((ones,X_train),axis=1)
X_train


# In[533]:


theta = np.zeros([15,12])
theta.shape


# In[534]:


def sigmoid(z):
    return 1 / (1 + np.exp(-z))
def grad_des(X_train,Y_train,thh,lrr,itr): 
    for i in range(itr):
        h = sigmoid(np.dot(X_train,thh.T))
        thh = thh - (lrr/len(X_train)) * np.sum(X_train * (h - Y_train), axis=0)  
    return thh


# In[ ]:





# In[535]:


for i in range(0,11):
    tt = np.zeros([1,12])
    W = np.array(Y_train)
    for j in range(len(W)):
        if W[j] == i:
            W[j] = 1
        else:
            W[j] = 0
    theta[i]=grad_des(X_train,W,tt,lrr,itr)
#     print(X_train.shape)


# In[536]:


y_pred = []
for index,rows in X_test.iterrows():
    rows = list(rows)
    m=0
    for a in range(0,11):
        y = 0
        for i in range(len(rows)):
            y = y + rows[i]*theta[a][i+1]
        y = y + theta[a][0]
        y = sigmoid(y)
        if y >= m:
            label = a
            m = y
    y_pred.append(label)
# y_pred


# In[539]:


# print((y_pred == Y_test).mean()*100)
print(confusion_matrix(Y_test,y_pred))
# print(classification_report(Y_test,y_pred))
print("Accuracy: ", accuracy_score(Y_test, y_pred)*100)


# In[538]:


sklearn()


# In[ ]:





# In[ ]:




