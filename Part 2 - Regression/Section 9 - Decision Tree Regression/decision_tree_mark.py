# -*- coding: utf-8 -*-
"""
Created on Mon Dec  3 13:40:28 2018

@author: Razer
"""

from sklearn.tree import DecisionTreeRegressor
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("Position_Salaries.csv")
X = data.iloc[:, 1:2].values
y = data.iloc[:, -1].values

regressor = DecisionTreeRegressor()
regressor.fit(X, y)


X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X, y, color='Green')
plt.plot(X_grid, regressor.predict(X_grid))
plt.title("Decision Tree Regressor (Predicting Salary)")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()
regressor.predict(6.5)