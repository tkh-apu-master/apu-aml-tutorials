# Importing the libraries
import numpy as np
# import matplotlib.pyplot as plt
import pandas as pd


# Importing the dataset
data = pd.read_csv('D:/APU/CT108-3-3 - OCDS/Lab Sessions/Lab3 - Data Preprocessing/Data.csv')
data_excel = pd.read_excel('D:/APU/CT108-3-3 - OCDS/Lab Sessions/Lab3 - Data Preprocessing/Loan approval.xlsx')
data
data.head()
data.shape
data.describe()
data.isnull().sum()
data["Country"].value_counts()

X = data.iloc[:, :-1].values
y = data.iloc[:, 3].values
X
y

# Replacing a catergorical variable using forward fill method | bfill for backward fill method
data["Country"].fillna(method = 'ffill', inplace = True)
data

# Replacing a catergorical variable with specific value
data["Country"].fillna("France", inplace = True)
data

# Rename the Area columnn to 'place_name'
data.rename(columns={"Age": "Person_Age"}, inplace=True)
data

# Drop the rows with the missing values
data.dropna()

# Drop the columns with the missing values
data.dropna(axis=1)

# Drop the rows where all columns have missing values
data.dropna(how='all')

# Dropping a specific column
data = data.drop(columns = 'Person_Age')
data

# Taking care of numeric missing data....value of the most common value (mode)
new_data = data.apply(lambda x: x.fillna(x.value_counts().index[0]))
new_data
new_data["Country"].value_counts()


# Taking care of numeric missing data
# strategies: ['mean', 'median', 'most_frequent', 'constant']
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')
X[:, 1:3] = imputer.fit_transform(X[:, 1:3])
X


from missingpy import KNNImputer
imputer = KNNImputer(n_neighbors=2)
X[:, 1:3] = imputer.fit_transform(X[:, 1:3])
X


from sklearn.experimental import enable_iterative_imputer  
from sklearn.impute import IterativeImputer
imputer = IterativeImputer(random_state=0)
X[:, 1:3] = imputer.fit_transform(X[:, 1:3])
X

'''
from sklearn.linear_model import BayesianRidge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor
'''
    
# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
X

onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()
X

# Encoding the Dependent Variable
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)
y

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
X_train
X_test
y_train
y_test

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

