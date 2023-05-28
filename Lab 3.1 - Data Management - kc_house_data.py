#!/usr/bin/env python
# coding: utf-8

# # Data Management - kc_house_data

# In[1]:


import pandas as pd
import numpy as np
import sklearn 


# In[2]:


df = pd.read_csv('D:/APU/CT108-3-3 - OCDS/Lab Sessions/Lab3 - Data Preprocessing/kc_house_data.csv')


# In[3]:


df.shape


# In[4]:


rows, cols = df.shape
print ('number of records:', rows)
print ('number of features:', cols)


# In[5]:


# pd.set_option('display.max_rows', None)
df


# In[6]:


df.head()


# In[7]:


df.tail()


# In[8]:


df.dtypes


# In[9]:


df.info


# In[10]:


df.describe() # Statistical Information of the data


# In[11]:


df['price'].describe()


# In[12]:


df = df.drop('id', axis=1)


# In[13]:


df.shape


# In[14]:


df.head()


# In[15]:


df = df.drop(df.columns[[0, 2, 3]], axis=1) 


# In[16]:


df.head(3)


# In[17]:


df.shape


# In[18]:


df = df.iloc[:, :-1]


# In[19]:


df.shape


# In[20]:


df.head(3)


# **Loading data from a website**

# In[21]:


target_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer/breast-cancer.data' 
df_bcancer = pd.read_csv(target_url, header=0, sep=",")


# In[22]:


df_bcancer.head()


# **Loading data from Sklearn**

# In[23]:


from sklearn.datasets import load_iris
iris = load_iris()


# In[24]:


n_samples, n_features = iris.data.shape
print ('number of samples:', n_samples)
print ('number of features:', n_features)
print (iris.data[0])


# In[25]:


iris.data.shape


# In[26]:


iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)


# In[27]:


iris_df.head


# In[ ]:




