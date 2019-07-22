#Polynomial Linear Regression
#Simple example showing how to create and plot a polynomial regression model.
#In the example code below, I'm not using any test/training sets.

#Also plotting a linear regression, for comparing linear - polynomial models.


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Importing the dataset
dataset = pd.read_csv('X.csv')
X = dataset.iloc[:, X:X].values
y = dataset.iloc[:, X].values

#Fitting the Linear Regression to the dataset.
lin_reg = LinearRegression()
lin_reg.fit(X,y)

#Fitting Polynomial Regression to the dataset.
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=4) #Set degree depending on the dimension.
X_poly = poly_reg.fit_transform(X)
lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly, y)

#Visualizing the Linear regression results.
#Using color = red for actual values.
plt.scatter(X, y, color='red')
#Using color = blue for predicted values.
plt.plot(X, lin_reg.predict(X), color='blue')
plt.title('Linear example.')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

#Visualizing the polynomial regression.
#Using color = red for actual values.
plt.scatter(X, y, color='red')
#Using color = blue for predicted values.
plt.plot(X, lin_reg2.predict(poly_reg.fit_transform(X)), color='blue')
plt.title('Polynomial example.')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()






















