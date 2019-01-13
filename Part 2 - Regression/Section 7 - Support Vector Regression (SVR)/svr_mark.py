# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 20:47:59 2018

@author: Razer
"""

import pandas as pd
import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures

data = pd.read_csv("Position_Salaries.csv")

X = data.iloc[:, 1:2].values
y = data.iloc[:, 2:].values

sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)

regressor = SVR(kernel='rbf')
regressor.fit(X, y)

y_pred = sc_y.inverse_transform(regressor.predict(sc_X.transform(np.array([[6.5]]))))

plt.scatter(X, y, color='red')
plt.plot(X, regressor.predict(X), color='blue')
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.title("SVR Salary prediction")
plt.show()