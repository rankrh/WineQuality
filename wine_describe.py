# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 11:54:08 2019

@author: Bob
"""

import numpy as np
import pandas as pd
import seaborn as sbn
import matplotlib.pyplot as plt

red_wine = pd.read_csv('winequality-red.csv', sep=';')
red_wine['type'] = 0

white_wine = pd.read_csv('winequality-white.csv', sep=';')
white_wine['type'] = 1

"""All wine, where type 0 is red, type 1 is white"""
all_wine = pd.concat([red_wine, white_wine])

print("Red")
print(red_wine.describe())
print()
print('White')
print(white_wine.describe())
print()
print("All")
print(all_wine.describe())

columns = list(red_wine.drop(columns=['type', 'quality']))
for column in columns:
    sbn.scatterplot(
        x=all_wine[column],
        y=all_wine.quality)


correlation_matrix = all_wine.corr(method="pearson")
sbn.heatmap(
    data=correlation_matrix,
    annot=True,
    linewidths=.5,)
plt.show()

sbn.countplot(red_wine.quality)
plt.title('Red Wine Quality')
plt.show()
print(pd.DataFrame(red_wine.quality.value_counts()).sort_index())

sbn.countplot(white_wine.quality)
plt.title("White Wine Quality")
plt.show()
print(pd.DataFrame(white_wine.quality.value_counts()).sort_index())

sbn.countplot(all_wine.quality)
plt.title("All Wine Quality")
plt.show()
print(pd.DataFrame(all_wine.quality.value_counts()).sort_index())

