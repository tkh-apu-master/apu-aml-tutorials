#!/usr/bin/env python
# coding: utf-8

# In[11]:


# Load libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor 
from sklearn import datasets
from IPython.display import Image  
from sklearn import tree
import pydotplus
from sklearn.model_selection import train_test_split 
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score


# In[2]:


df = pd.read_csv("D:/APU/CT108-3-3 - OCDS/Lab Sessions/Lab9 - DT/kc_house_data.csv")
df.head()


# In[3]:


data = df.drop(['id', 'date', 'yr_renovated', 'zipcode', 'lat', 'long'], axis=1)
data.head()


# In[4]:


data.isnull().sum()


# In[5]:


X = data.drop('price', axis=1)
y = data['price']


# In[6]:


# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1) 


# In[7]:


# Create decision tree classifer object
dt = DecisionTreeRegressor()
dt.fit(X_train, y_train)

y_pred = dt.predict(X_test)

print(dt.get_params())


# In[8]:


print("R-Squared on train dataset = {}".format(dt.score(X_train, y_train).round(3)))
print("R-Squared on test dataset = {}".format(dt.score(X_test, y_test).round(3)))


# In[9]:


df2 = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
df2.head()


# In[12]:


from sklearn.metrics import mean_squared_error, mean_absolute_error
print("Mean Squared Error: ", mean_squared_error(y_test, y_pred).round(2))
print('Mean Absolute Error: ', mean_absolute_error(y_test, y_pred).round(2))  
print('Root Mean Squared Error: ', np.sqrt(mean_squared_error(y_test, y_pred)).round(2))


# **Important Features**

# In[13]:


import matplotlib.pyplot as plt
feature_imp = pd.Series(dt.feature_importances_, index=X_train.columns).sort_values(ascending=False)

# Creating a bar plot, displaying only the top k features
sns.barplot(x=feature_imp[:15], y=feature_imp.index[:15])
# Add labels to your graph
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Important Features")
plt.show()


# **Grid Search CV**

# In[24]:


dt = DecisionTreeRegressor()

# Hyperparameter Optimization
parameters = {'max_features': ['log2', 'sqrt','auto'],
              'criterion': ['mse', 'mae'],
              'max_depth': [2, 3, 5, 10, 50], 
              'max_leaf_nodes': [5, 20, 100],
             }

# Run the grid search
grid_search_dt = GridSearchCV(dt, parameters, cv=5)

grid_search_dt.fit(X_train,y_train)
best_parameters_dt = grid_search_dt.best_params_  
best_score_dt = grid_search_dt.best_score_ 
print(best_parameters_dt)
print(best_score_dt)

y_pred_1 = grid_search_dt.predict(X_test)

print("Mean Squared Error: ", mean_squared_error(y_test, y_pred_1).round(2))
print('Mean Absolute Error: ', mean_absolute_error(y_test, y_pred_1).round(2))  
print('Root Mean Squared Error: ', np.sqrt(mean_squared_error(y_test, y_pred_1)).round(2))


# In[25]:


df3 = pd.DataFrame({'Actual': y_test, 'Old Predicted': y_pred, 'New Predicted': y_pred_1.round(2)})
df3.head()


# In[ ]:




