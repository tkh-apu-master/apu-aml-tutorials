#!/usr/bin/env python
# coding: utf-8

# In[36]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.svm import SVC
import warnings
warnings.filterwarnings("ignore")


# In[2]:


url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

# Assign colum names to the dataset
colnames = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']

# Read dataset to pandas dataframe
irisdata = pd.read_csv(url, names=colnames)


# In[3]:


irisdata


# In[4]:


irisdata.head(10)


# In[5]:


irisdata.shape


# In[6]:


irisdata.describe()


# In[7]:


irisdata.isnull().sum()


# In[8]:


irisdata['Class'].value_counts()


# In[9]:


#class distribution 
irisdata['Class'].value_counts()
sns.countplot(x='Class',data=irisdata,palette='hls')
plt.show()
plt.savefig('Class')


# In[10]:


irisdata.hist(figsize=(20,10), edgecolor="powderblue", color="powderblue")
plt.show()


# In[11]:


# Plotting Petal Length vs Petal Width & Sepal Length vs Sepal width
# warnings.simplefilter("ignore")
# Supress any warning
plt.figure()
fig,ax=plt.subplots(1,2,figsize=(17, 9))
irisdata.plot(x="sepal-length",y="sepal-width",kind="scatter",ax=ax[0],sharex=False,sharey=False,label="sepal",color='r')
irisdata.plot(x="petal-length",y="petal-width",kind="scatter",ax=ax[1],sharex=False,sharey=False,label="petal",color='b')
ax[0].set(title='Sepal comparasion ', ylabel='sepal-width')
ax[1].set(title='Petal Comparasion',  ylabel='petal-width')
ax[0].legend()
ax[1].legend()
# plt.show()
# plt.close()

# we can see that  there are some petals which are smaller than rest of petal.
#Let's examine them


# In[12]:


# for each Species ,let's check what is petal and sepal distibutuon
plt.figure()
fig,ax=plt.subplots(1,2,figsize=(21, 10))

irisdata[irisdata['Class']=='Iris-setosa'].plot(x="sepal-length", y="sepal-width", kind="scatter",ax=ax[0],label='Iris-setosa',color='r')
irisdata[irisdata['Class']=='Iris-versicolor'].plot(x="sepal-length",y="sepal-width",kind="scatter",ax=ax[0],label='Iris-versicolor',color='b')
irisdata[irisdata['Class']=='Iris-virginica'].plot(x="sepal-length", y="sepal-width", kind="scatter", ax=ax[0], label='Iris-virginica', color='g')

irisdata[irisdata['Class']=='Iris-setosa'].plot(x="petal-length", y="petal-width", kind="scatter",ax=ax[1],label='Iris-setosa',color='r')
irisdata[irisdata['Class']=='Iris-versicolor'].plot(x="petal-length",y="petal-width",kind="scatter",ax=ax[1],label='Iris-versicolor',color='b')
irisdata[irisdata['Class']=='Iris-virginica'].plot(x="petal-length", y="petal-width", kind="scatter", ax=ax[1], label='Iris-virginica', color='g')

ax[0].set(title='Sepal comparasion ', ylabel='sepal-width')
ax[1].set(title='Petal Comparasion',  ylabel='petal-width')
ax[0].legend()
ax[1].legend()

# plt.show()
# plt.close()

# satosa   - satosa Petal are relatively smaller than rest of species .can be easily separable from rest of Species 
# versicolor & virginica are also separable in Petal comprasion
# satoa sepal are smallest in length and largest in Width than other species


# In[13]:


X = irisdata.drop('Class', axis=1)
y = irisdata['Class']


# In[14]:


X.head()


# In[15]:


y.head()


# In[16]:


sns.pairplot(data=irisdata, hue='Class', palette='Set2')


# In[17]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)


# In[18]:


X_test


# In[19]:


y_test


# **Using Linear Kernel**

# In[42]:


svclassifier = SVC(kernel='linear', gamma='scale')
svclassifier.fit(X_train, y_train)


# In[43]:


y_pred_lnr = svclassifier.predict(X_test)


# In[44]:


y_pred_lnr


# In[46]:


from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
print("Linear Kernal of SVC")
print("Accuracy = ", accuracy_score(y_test, y_pred_lnr))
print("\nConfusion Matrix")
print(confusion_matrix(y_test, y_pred_lnr))
print("\nClassification Report")
print(classification_report(y_test, y_pred_lnr))


# **Using Polynomial Kernel**

# In[47]:


svclassifier = SVC(kernel='poly', degree=8, gamma='scale')
svclassifier.fit(X_train, y_train)


# In[48]:


y_pred_poly = svclassifier.predict(X_test)


# In[49]:


y_pred_poly


# In[50]:


from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
print("Polynomial Kernal of SVC")
print("Accuracy = ", accuracy_score(y_test, y_pred_poly))
print("\nConfusion Matrix")
print(confusion_matrix(y_test, y_pred_poly))
print("\nClassification Report")
print(classification_report(y_test, y_pred_poly))


# **Using RBF Kernel**

# In[64]:


svclassifier = SVC(kernel='rbf', gamma='scale')
svclassifier.fit(X_train, y_train)


# In[65]:


y_pred_rbf = svclassifier.predict(X_test)


# In[66]:


y_pred_rbf


# In[67]:


print("RBF Kernal of SVC")
print("Accuracy = ", accuracy_score(y_test, y_pred_rbf))
print("\nConfusion Matrix")
print(confusion_matrix(y_test, y_pred_rbf))
print("\nClassification Report")
print(classification_report(y_test, y_pred_rbf))


# In[68]:


# Using Sigmoid Kernel
svclassifier = SVC(kernel='sigmoid', gamma='scale')
svclassifier.fit(X_train, y_train)


# In[69]:


y_pred_sig = svclassifier.predict(X_test)


# In[70]:


y_pred_sig


# In[71]:


print("Sigmoid Kernal of SVC")
print("Accuracy = ", accuracy_score(y_test, y_pred_sig))
print("\nConfusion Matrix")
print(confusion_matrix(y_test, y_pred_sig))
print("\nClassification Report")
print(classification_report(y_test, y_pred_sig))


# Comparison of Kernel Performance.
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# If we compare the performance of the different types of kernels we can clearly see that the sigmoid kernel performs the worst. This is due to the reason that sigmoid function returns two values, 0 and 1, therefore it is more suitable for binary 
# classification problems. However, in our case we had three output classes.
# 
# However, there is no hard and fast rule as to which kernel performs best in every scenario. It is all about testing all the kernels and selecting the one with the best results on your test dataset.

# **Grid Search with CV**

# In[75]:


from sklearn.model_selection import GridSearchCV
from numpy import arange
model_svc= SVC()


parameters = dict()
parameters['kernel'] = ['rbf', 'poly', 'linear', 'sigmoid']
parameters['C'] = arange(1, 1000, 10)
parameters['gamma'] = arange(1e-4, 1, 10)
parameters['degree'] = arange(1, 10, 1)
parameters['class_weight'] = ['dict', 'balanced']
parameters['random_state'] = arange(1, 10, 1)

## Building Grid Search algorithm with cross-validation and acc score.

grid_search_svc = GridSearchCV(estimator=model_svc, param_grid=parameters, scoring='accuracy', cv=5, n_jobs=-1)

grid_search_svc.fit(X_train,y_train)
best_parameters_svc = grid_search_svc.best_params_  
best_score_svc = grid_search_svc.best_score_ 
print(best_parameters_svc)
print(best_score_svc)

y_pred=grid_search_svc.predict(X_test)

# Get the accuracy score
svc_acc=accuracy_score(y_test, y_pred)*100
svc_pre=precision_score(y_test, y_pred, average='micro')
svc_recall=recall_score(y_test, y_pred, average='micro')
svc_f1_=f1_score(y_test, y_pred, average='micro')

print("SVM - Accuracy: {:.3f}.".format(svc_acc))
print("SVM - Precision: {:.3f}.".format(svc_pre))
print("SVM - Recall: {:.3f}.".format(svc_recall))
print("SVM - F1_Score: {:.3f}.".format(svc_f1_))


# In[77]:


print("\nConfusion Matrix")
print(confusion_matrix(y_test, y_pred))


# In[76]:


print("\nClassification Report")
print(classification_report(y_test, y_pred))

