# -*- coding: utf-8 -*-
"""
Created on Mon Dec  3 18:20:25 2018

@author: Razer
"""
"""importing libraries"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from matplotlib.colors import ListedColormap

"""convert data into a DataFrame"""
data = pd.read_csv("Social_Network_Ads.csv")

"""Store Features on X and the ^P value on y"""
X = data.iloc[:, [2,3]].values
y = data.iloc[:, 4].values

"""Split X and y into train and test data set"""
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, random_state=0)

"""Scale training and test features, 
It is too big then: (Convert it to a reasonable range of value)"""
X_scale = StandardScaler()
X_train = X_scale.fit_transform(X_train)
X_test = X_scale.transform(X_test)

"""Create our logistic regression model then fit it to our training data set"""
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)

"""Predict our test set on our model"""
y_pred = classifier.predict(X_test)

"""Create a confusion matrix to show correctly/incorrectly identified values"""
cm = confusion_matrix(y_test, y_pred)

"""Assign to new variable for an easy use on future"""
X_set, y_set = X_test, y_test

"""Visualizing the data

   Make a meshgrid to for a higher resolution of data"""
X1,X2 = np.meshgrid(np.arange(start=X_set[:, 0].min()-1, stop=X_set[:, 0].max()+1, step=0.01),
                    np.arange(start=X_set[: 1].min()-1, stop=X_set[:, 1].max()+1, step=0.01))

"""Plot our model in a contour"""
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             cmap=ListedColormap(('red', 'green')), alpha=0.40)
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

"""Plot our test data in the model"""
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j,0], X_set[y_set == j, 1],
                c=ListedColormap( ('red','green') )(i), label=j)
plt.xlabel("Age")
plt.ylabel("Estimated Salary")
plt.legend()
plt.title("Logistic Regression (Test set)")
#plt.savefig("Logistic Regression (Test set)")
plt.show()