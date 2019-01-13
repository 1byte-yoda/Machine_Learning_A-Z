# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 14:30:09 2018

@author: Razer
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import datasets

boston = datasets.load_boston()
dataset = pd.read_csv("Salary_Data.csv")

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=1/3)

regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

plt.figure(1)
plt.scatter(X_train, y_train, color='green')
plt.plot(X_train, regressor.predict(X_train), color='black')
plt.title("Salary vs. Experience (Training set)")
plt.xlabel("Experience")
plt.ylabel("Salary")
plt.show()

plt.figure(2)
plt.scatter(X_test, y_test, color='green')
plt.plot(X_train, regressor.predict(X_train), color='black')
plt.title("Salary vs. Experience (Test set)")
plt.xlabel("Experience")
plt.ylabel("Salary")
plt.show()