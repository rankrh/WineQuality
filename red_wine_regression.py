# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 10:35:25 2019

@author: Bob
"""

import numpy as np
import pandas as pd
import seaborn as sbn
import matplotlib.pyplot as plt

from sklearn.cross_validation import train_test_split
from sklearn import preprocessing


red_wine = pd.read_csv('winequality-red.csv', sep=';')

print(red_wine.head())
print(red_wine.describe())

correlation_matrix = red_wine.corr(method="pearson")
sbn.heatmap(
    data=correlation_matrix,
    annot=True,
    linewidths=.5,)
plt.show()

x = red_wine.drop('quality', axis=1)
y = red_wine.quality

x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.2,
    random_state=108,
    stratify=y)

scaler = preprocessing.StandardScaler().fit(x_train)
x_train_scaled = scaler.transform(x_train)
x_test_scaled = scaler.transform(x_test)
print(x_test_scaled.mean(axis=0))
