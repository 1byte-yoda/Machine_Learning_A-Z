# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 13:02:33 2018

@author: Mark
"""
"""Import libraries """
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import Imputer, OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

"""Read the data"""
df = pd.read_csv("Data.csv")
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

"""Deal with missing values, fill with the mean of the column with the NaN value"""
X_imputer = Imputer(missing_values=np.NaN, strategy="mean", axis=0)
X_imputer = X_imputer.fit(X[:, 1:3])
X[:, 1:3] = X_imputer.transform(X[:, 1:3])

"""Encode our dependent variable (country)"""
label_encoder_X = LabelEncoder()
X[:, 0] = label_encoder_X.fit_transform(X[:, 0])

"""Encode country into n columns, that is based
   on how many categories are there in the country column"""
one_hot_encoder_X = OneHotEncoder(categorical_features=[0])
X = one_hot_encoder_X.fit_transform(X).toarray()

"""Encode our dependent categorical variable"""
label_encoder_y = LabelEncoder()
y = label_encoder_y.fit_transform(y)

"""Split data into training and test data set"""
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state=40)

"""Scale datas so that both columns/variables has the same range/scale (Feature Scaling)"""
sc_X = StandardScaler(with_mean=True, with_std=False)
X_tr = sc_X.fit_transform(X_train)
X_te = sc_X.transform(X_test)