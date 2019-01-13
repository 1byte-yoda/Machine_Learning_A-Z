# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 12:37:49 2018

@author: Razer
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

data = pd.read_csv("Position_Salaries.csv")

X = data.iloc[:, 1:2].values
y = data.iloc[:, 2].values

lin_reg = LinearRegression()
lin_reg.fit(X, y)

poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(X)
poly_reg.fit(X_poly)
lin_reg_poly = LinearRegression()
lin_reg_poly.fit(X_poly, y)

plt.figure(1)
plt.scatter(X, y, color='Green')
plt.plot(X, lin_reg.predict(X))
plt.title("Salary Prediction (Simple Linear Regression)")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()

plt.figure(2)
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((X_grid.size, 1))
plt.scatter(X, y, color="Green")
plt.plot(X, lin_reg_poly.predict(poly_reg.fit_transform(X)), color="black")
plt.title("Salary Prediction (Polynomial Linear Regression)")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()

predictions = dict()

predictions["Simple Linear Regression"] = lin_reg.predict(6.5)
predictions["Polynomial Linear Regression"] = lin_reg_poly.predict(poly_reg.fit_transform(6.5))

print("Salary Prediction for level 6.5 position")
for salary_pred in predictions.keys():
    print("{0}: ${1:,.2f}".format(salary_pred, predictions[salary_pred][0]))