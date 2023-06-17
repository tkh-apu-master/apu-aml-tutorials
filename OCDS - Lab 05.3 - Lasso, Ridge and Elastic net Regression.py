#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


Advertising = pd.read_csv("D:/APU/CT108-3-3 - OCDS/Lab Sessions/Lab5 - Linear Regression/Advertising.csv")
Advertising.head()


# In[3]:


Advertising.columns


# In[4]:


Advertising.shape


# In[5]:


Advertising.isnull().sum()


# In[6]:


Advertising.describe()


# In[7]:


def scatter_plot(feature, target):
    plt.figure(figsize=(16, 8))
    plt.scatter(
        Advertising[feature],
        Advertising[target],
        c='blue'
    )
    plt.xlabel("Money spent on {} ads ($)".format(feature))
    plt.ylabel("Sales ($k)")
    plt.show()


# In[8]:


scatter_plot('TV', 'sales')


# In[9]:


scatter_plot('radio', 'sales')


# In[10]:


scatter_plot('newspaper', 'sales')


# # Modelling

# ### Multiple Linear Regression - Ordinary Least Squares fitting

# In[11]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import cross_val_score
import statsmodels.api as sm

X = Advertising.drop(['sales'], axis=1)
y = Advertising['sales'].values.reshape(-1,1)

lin_reg = LinearRegression()
fit = lin_reg.fit(X, y)
print(fit)

X2 = sm.add_constant(X)
est = sm.OLS(y, X2)
est2 = est.fit()
print(est2.summary())


# In[12]:


print("Intercept: ", lin_reg.intercept_)
print("Coefficient: ", lin_reg.coef_)


# In[13]:


# Predictions
y_pred = lin_reg.predict(X)
print("~~~~Pedicted Sales Values~~~~")
print(y_pred)


# **Coefficient of Determination**

# In[14]:


print("R Square value:", r2_score(y, y_pred))


# In[15]:


R2s = cross_val_score(lin_reg, X, y, cv=5, scoring='r2')
print(R2s)
print("Average R Square after CV: ", np.mean(R2s))


# **Mean Squared Error**

# In[16]:


print("Mean Squared Error: ", mean_squared_error(y_pred, y))


# In[17]:


MSEs = cross_val_score(lin_reg, X, y, cv=5, scoring='neg_mean_squared_error')
print(MSEs)
print("Average MSE after CV: ", np.mean(MSEs))


# ### Ridge regression ---------------------------------

# **Ridge Regression performs ‘L2 regularization‘**

# Ridge regression is a variant of linear regression. The term above is the ridge constraint to the OLS equation

# Ridge regression is a way to create a parsimonious model when the number of predictor variables in a set exceeds the number of observations, or when a data set has multicollinearity (correlations between predictor variables).

# In[18]:


from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge
from numpy import arange

ridge = Ridge(solver = 'saga') # Stochastic Average Gradient descent which will set the step size because the solver computes 
# the step size (learning Rate) based on your data and alpha.

ridge_parameters = dict()
ridge_parameters['alpha'] = arange(1e-5, 100.0, 10)

ridge_regressor = GridSearchCV(ridge, ridge_parameters, scoring='neg_mean_squared_error', cv=5)

ridge_regressor.fit(X, y)


# **Best_params:** Parameter setting that gave the best results on the hold out data.

# In[19]:


ridge_regressor.best_params_


# **Best_score:** Mean cross-validated score of the best_estimator. Usually MSE given in negative value.

# In[20]:


ridge_regressor.best_score_


# ### Lasso Regression ---------------------------------

# **Ridge Regression performs ‘L1 regularization‘**

# Lasso Regression, which penalizes the sum of absolute values of the coefficients (L1 penalty).

# Useful when a large number of features are involved as Lasso will eliminate many features, and reduce overfitting in the linear model.

# In[21]:


from sklearn.linear_model import Lasso
from numpy import arange

lasso = Lasso()

lasso_parameters = dict()
lasso_parameters['alpha'] = arange(1e-5, 100.0, 10)

lasso_regressor = GridSearchCV(lasso, lasso_parameters, scoring='neg_mean_squared_error', cv = 5, n_jobs=-1)

lasso_regressor.fit(X, y)


# **Best_params:** Parameter setting that gave the best results on the hold out data.

# In[22]:


lasso_regressor.best_params_


# **Best_score:** Mean cross-validated score of the best_estimator. Usually MSE given in negative value.

# In[23]:


lasso_regressor.best_score_


# ### Elastic Net Regression ---------------------------------

# Elastic net is a popular type of regularized linear regression that combines two popular penalties, specifically the L1 and L2 penalty functions.

# In[24]:


from sklearn.linear_model import ElasticNet
from numpy import arange

e_net = ElasticNet(alpha=1.0, l1_ratio=0.5)

e_net_parameters = dict()
e_net_parameters['alpha'] = arange(1e-5, 100.0, 10)
e_net_parameters['l1_ratio'] = arange(0, 1, 0.01)

e_net_regressor = GridSearchCV(e_net, e_net_parameters, scoring='neg_mean_squared_error', cv = 5, n_jobs=-1)

e_net_regressor.fit(X, y)


# **Best_params:** Parameter setting that gave the best results on the hold out data.

# In[25]:


e_net_regressor.best_params_


# **Best_score:** Mean cross-validated score of the best_estimator. Usually MSE given in negative value.

# In[26]:


e_net_regressor.best_score_

