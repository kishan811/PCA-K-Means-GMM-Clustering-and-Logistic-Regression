#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy


# In[4]:


df = pd.read_csv('intrusion_detection/Train_data.csv')


# In[5]:


df.head()


# ### PCA doesn't seem to work well with categorical data. 
# ### Doing regular PCA on raw categorical values is not recommended.
# 

# ### Because PCA uses Covariance Matrix which uses Mean and variance like numerical values. <br>
# ### And finding mean and variance/ Std deviation for categorical attributes, by converting them into numerical integers, doesn't give any relevant information. So, it is not suggested to use PCA for categorical data.
# <br>
# 
#     

# #### But, there are some ways to do PCA on categorical:- <br><br>
#      1) Ordinal PCA: 
#         Ignoring all categorical discreteness.
#      2) Group means method:
#         By calculating some score.
#  etc etc.

# In[ ]:




