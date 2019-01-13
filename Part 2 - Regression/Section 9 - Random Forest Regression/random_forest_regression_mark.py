# -*- coding: utf-8 -*-
"""
Created on Mon Dec  3 16:49:12 2018

@author: Razer
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

data = pd.read_csv("Position_Salaries.csv")
X = data.iloc[:, 1:2].values
y = data.iloc[:, -1].values

regressor = RandomForestRegressor(n_estimators = 250, random_state=0)
regressor.fit(X,y)

y_pred = regressor.predict(6.5)

X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X, y, color='red')
plt.plot(X_grid, regressor.predict(X_grid), color='green')
plt.title("Random Forest Regression (Salary Prediction)")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show()