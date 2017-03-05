# -*- coding: utf-8 -*-
"""
Created on Sun Mar  5 10:27:24 2017

@author: mje
"""

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import permutation_test_score

for est in gat.estimators_:
    for tmp in est:
        lr_mean = LogisticRegression(C=0.0001)
        lr_mean.coef_ = np.asarray([lr.coef_ for lr in est]).mean(axis=0).squeeze()
        lr_mean.intercept_ = np.asarray([lr.intercept_ for lr in est]).mean()

            