#!/usr/bin/env python
# coding: utf-8

# # Polynomial Regression

# Polynomial regression is a form of regression analysis in which the relationship between the independent variable x and the dependent variable y is modelled as an nth degree polynomial in x. 

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import r2_score


# In[2]:


# Importing the dataset
dataset = pd.read_csv('D:/APU/CT108-3-3 - OCDS/Lab Sessions/Lab5 - Linear Regression/Position_Salaries.csv')
dataset


# ![image.png](attachment:image.png)

# In[3]:


X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values


# In[4]:


X


# In[5]:


y


# In[6]:


# Fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)


# In[7]:


# Visualizing the Linear Regression results
def viz_linear():
    plt.scatter(X, y, color='red')
    plt.plot(X, lin_reg.predict(X), color='blue')
    plt.title('Truth or Bluff (Linear Regression)')
    plt.xlabel('Position level')
    plt.ylabel('Salary')
    plt.show()
    return
viz_linear()


# In[8]:


# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(X)
pol_reg = LinearRegression()
pol_reg.fit(X_poly, y)


# **To get an overview of the increment of salary, let’s visualize the data set into a chart**

# In[9]:


# Visualizing the Polymonial Regression results
def viz_polymonial():
    plt.scatter(X, y, color='red')
    plt.plot(X, pol_reg.predict(poly_reg.fit_transform(X)), color='blue')
    plt.title('Truth or Bluff (Linear Regression)')
    plt.xlabel('Position level')
    plt.ylabel('Salary')
    plt.show()
    return
viz_polymonial()


# In[10]:


# Predicting a new result with Linear Regression
print(lin_reg.predict(X).round())
print("R Square value:", r2_score(y, lin_reg.predict(X).round()))


# In[11]:


# Predicting a new result with Polymonial Regression
print(pol_reg.predict(poly_reg.fit_transform(X)).round())
print("R Square value:", r2_score(y, pol_reg.predict(poly_reg.fit_transform(X)).round()))

