# Multiple Linear Regression.
#Showing a simple Multiple Linear regression model, and how to find the variables needed.

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('X.csv')
X = dataset.iloc[:, :X].values
y = dataset.iloc[:, X].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
X[:, X] = labelencoder.fit_transform(X[:, X])
onehotencoder = OneHotEncoder(categorical_features = [X])
X = onehotencoder.fit_transform(X).toarray()

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

#Starting backward eliminination.
X = np.append(arr = np.ones((50, 1)).astype(int), values = X,axis = 1)

import statsmodels.formula.api as sm
#Creating my optimal matrix, with all independeble variables.
X_opt = X[:,[0, 1, 2, 3, 4, 5]]
#Fitting my X_opt matrix to find the p values of the indipendle variable.
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
#Removing independeble variables that has too high p-value.
X_opt = X[:,[0, 3,]]
#Fitting my X_opt matrix to find the p values of the indipendle variable.
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()
