#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 19:22:01 2020

@author: liubei
"""


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('/home/liubei/Desktop/deep-learnning-course/ANN/P16-Artificial-Neural-Networks/Artificial_Neural_Networks/Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

test_y = np.array([[600, 'France', 'Male', 40, 3, 60000, 2, 1, 1, 50000]])
test_y_2 = np.array([[600, 0, 1, 40, 3, 60000, 2, 1, 1, 50000]])

# Encoding categorical data
from sklearn.compose import ColumnTransformer 
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
ct = ColumnTransformer([('Geography', OneHotEncoder(), [1])], remainder='passthrough')
X = ct.fit_transform(X)
X = X[:, 1:]


#abelencoder_y_test_1 = LabelEncoder()
test_y[:, 1] = labelencoder_X_1.transform(test_y[:, 1])
#abelencoder_y_test_2 = LabelEncoder()
test_y[:, 2] = labelencoder_X_2.transform(test_y[:, 2])

test_y_arr = []
for arr in test_y:
    for j in arr:
        test_y_arr.append(int(j))

test_y_encoded = ct.transform(np.array([test_y_arr]))