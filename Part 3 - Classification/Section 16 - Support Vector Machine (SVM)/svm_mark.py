# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 17:34:41 2018

@author: Razer
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from matplotlib.colors import ListedColormap
data = pd.read_csv("Social_Network_Ads.csv")
X = data.iloc[:, [2,3]].values
y = data.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

X_sc = StandardScaler()
X_train = X_sc.fit_transform(X_train)
X_test = X_sc.transform(X_test)

#classifier = SVC(kernel="linear", random_state=0)
classifier = SVC(kernel="rbf", random_state=0)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

cm = confusion_matrix(y_test, y_pred)

X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max()+1, step = 0.01),
                     np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max()+1, step = 0.01))

plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             cmap = ListedColormap(('red', 'green')), alpha=0.40)

plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_test == j,0], X_set[y_test == j,1],  c=ListedColormap(('red','green'))(i),
                label=j)
plt.show()