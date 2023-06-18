#!/usr/bin/env python
# coding: utf-8

# # SVM - Binary

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")


# In[2]:


'''from google.colab import files
file = files.upload()
import io
data = pd.read_csv(io.BytesIO(file['diabetes.csv']))'''


# In[3]:


data = pd.read_csv('D:/APU/CT108-3-3 - OCDS/Lab Sessions/Lab8 - SVM/diabetes.csv')


# In[4]:


data.head()


# In[5]:


data.shape


# In[6]:


data.describe()


# In[7]:


data.isnull().sum()


# In[8]:


data['Outcome'].value_counts()


# In[9]:


# class distribution 
data['Outcome'].value_counts()
sns.countplot(x='Outcome', data=data, palette='hls')
plt.show()
plt.savefig('Outcome')


# In[10]:


X = data.drop('Outcome', axis=1)
y = data['Outcome']


# In[11]:


sc_X = StandardScaler()
X = sc_X.fit_transform(X)


# In[12]:


y.head()


# In[13]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state=0)


# In[14]:


X_test[0:5]


# In[15]:


y_test.head()


# **SVC - Basic**

# In[16]:


from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
svc= SVC()
svc.fit(X_train, y_train)
y_p = svc.predict(X_test)
acc=accuracy_score(y_test, y_p)*100
print("SVM - Accuracy: {:.3f}.".format(acc))
print("\nClassification Report")
print(classification_report(y_test, y_p))


# **Grid Search CV without class balancing**

# Grid Search sets up a grid of hyperparameter values and for each combination, trains a model and scores on the testing data. 

# In[17]:


from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from numpy import arange
grid_svc= SVC()

parameters = dict()
parameters['kernel'] = ['rbf', 'poly', 'linear', 'sigmoid']
parameters['C'] = arange(1, 10, 1)
parameters['gamma'] = ['scale', 'auto']
parameters['class_weight'] = ['dict', 'balanced']

## Building Grid Search algorithm with cross-validation and acc score.

grid_search_svc = GridSearchCV(grid_svc, parameters, scoring='accuracy', cv=5, n_jobs=-1)

grid_search_svc.fit(X_train,y_train)
best_parameters_svc = grid_search_svc.best_params_  
best_score_svc = grid_search_svc.best_score_ 
print(best_parameters_svc)
print(best_score_svc)

y_pred = grid_search_svc.predict(X_test)

# Get the accuracy score
svc_acc=accuracy_score(y_test, y_pred)*100
svc_pre=precision_score(y_test, y_pred, average='micro')
svc_recall=recall_score(y_test, y_pred, average='micro')
svc_f1_=f1_score(y_test, y_pred, average='micro')

print("\nSVM - Accuracy: {:.3f}.".format(svc_acc))
print("SVM - Precision: {:.3f}.".format(svc_pre))
print("SVM - Recall: {:.3f}.".format(svc_recall))
print("SVM - F1_Score: {:.3f}.".format(svc_f1_))
print("\nClassification Report")
print(classification_report(y_test, y_pred))


# **Random Search CV without class balancing**

# Random Search sets up a grid of hyperparameter values and selects random combinations to train the model and score. 

# In[18]:


from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from numpy import arange
rand_svc= SVC()

parameters = dict()
parameters['kernel'] = ['rbf', 'poly', 'linear', 'sigmoid']
parameters['C'] = arange(1, 10, 1)
parameters['gamma'] = ['scale', 'auto']
parameters['class_weight'] = ['dict', 'balanced']

## Building Grid Search algorithm with cross-validation and acc score.

rand_search_svc = RandomizedSearchCV(rand_svc, parameters, scoring='accuracy', cv=5, n_jobs=-1)

rand_search_svc.fit(X_train,y_train)
best_parameters_svc = rand_search_svc.best_params_  
best_score_svc = rand_search_svc.best_score_ 
print(best_parameters_svc)
print(best_score_svc)

y_pred = rand_search_svc.predict(X_test)

# Get the accuracy score
svc_acc=accuracy_score(y_test, y_pred)*100
svc_pre=precision_score(y_test, y_pred, average='micro')
svc_recall=recall_score(y_test, y_pred, average='micro')
svc_f1_=f1_score(y_test, y_pred, average='micro')

print("\nSVM - Accuracy: {:.3f}.".format(svc_acc))
print("SVM - Precision: {:.3f}.".format(svc_pre))
print("SVM - Recall: {:.3f}.".format(svc_recall))
print("SVM - F1_Score: {:.3f}.".format(svc_f1_))
print("\nClassification Report")
print(classification_report(y_test, y_pred))


# **Class balancing**

# In[19]:


from collections import Counter
from imblearn.over_sampling import SMOTE
X_b, y_b = SMOTE().fit_resample(X, y)

plt.subplots(figsize=(5,5))
sns.countplot(x=y_b)
print(Counter(y_b))


# In[20]:


# Split the dataset into a test and training set
X_tr, X_te, y_tr, y_te = train_test_split(X_b, y_b, test_size=0.2, random_state=0)


# **Grid Search CV with class balancing**

# In[21]:


from sklearn.model_selection import GridSearchCV
from numpy import arange
model_svc= SVC()


parameters = dict()
parameters['kernel'] = ['rbf', 'poly', 'linear', 'sigmoid']
parameters['C'] = arange(1, 10, 1)
parameters['gamma'] = ['scale', 'auto']
parameters['class_weight'] = ['dict', 'balanced']

## Building Grid Search algorithm with cross-validation and acc score.

grid_search_svc_2 = GridSearchCV(model_svc, parameters, scoring='accuracy', cv=5, n_jobs=-1)

## Lastly, finding the best parameters.
grid_search_svc_2.fit(X_tr, y_tr)
best_parameters_SVC_2 = grid_search_svc_2.best_params_  
best_score_SVC_2 = grid_search_svc_2.best_score_ 
print()
print(best_parameters_SVC_2)
print(best_score_SVC_2)

y_pred_2 = grid_search_svc_2.predict(X_te)

# Get the accuracy score
svc_acc_2 = accuracy_score(y_te, y_pred_2)*100
svc_pre_2 = precision_score(y_te, y_pred_2, average='micro')
svc_recall_2 = recall_score(y_te, y_pred_2, average='micro')
svc_f1_2 = f1_score(y_te, y_pred_2, average='micro')

print("\nSVM - Accuracy: {:.3f}.".format(svc_acc_2))
print("SVM - Precision: {:.3f}.".format(svc_pre_2))
print("SVM - Recall: {:.3f}.".format(svc_recall_2))
print("SVM - F1 Score: {:.3f}.".format(svc_f1_2))
print ('\n Clasification Report:\n', classification_report(y_te,y_pred_2))


# **Random Search CV with class balancing**

# In[22]:


from sklearn.model_selection import RandomizedSearchCV
from numpy import arange
model_svc= SVC()

parameters = dict()
parameters['kernel'] = ['rbf', 'poly', 'linear', 'sigmoid']
parameters['C'] = arange(1, 10, 1)
parameters['gamma'] = ['scale', 'auto']
parameters['class_weight'] = ['dict', 'balanced']

## Building Random Search algorithm with cross-validation and acc score.

rand_search_svc_2 = RandomizedSearchCV(model_svc, parameters, scoring='accuracy', cv=5, n_jobs=-1)

## Lastly, finding the best parameters.
rand_search_svc_2.fit(X_tr, y_tr)
best_parameters_SVC_2 = rand_search_svc_2.best_params_  
best_score_SVC_2 = rand_search_svc_2.best_score_ 
print()
print(best_parameters_SVC_2)
print(best_score_SVC_2)

y_pred_2 = rand_search_svc_2.predict(X_te)

# Get the accuracy score
lr_acc_2 = accuracy_score(y_te, y_pred_2)*100
lr_pre_2 = precision_score(y_te, y_pred_2, average='micro')
lr_recall_2 = recall_score(y_te, y_pred_2, average='micro')
lr_f1_2 = f1_score(y_te, y_pred_2, average='micro')

print("\nSVM - Accuracy: {:.3f}.".format(lr_acc_2))
print("SVM - Precision: {:.3f}.".format(lr_pre_2))
print("SVM - Recall: {:.3f}.".format(lr_recall_2))
print("SVM - F1 Score: {:.3f}.".format(lr_f1_2))
print ('\n Clasification Report:\n', classification_report(y_te,y_pred_2))

