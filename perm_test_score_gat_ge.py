# -*- coding: utf-8 -*-
"""
Created on Sun Mar  5 10:27:24 2017

@author: mje
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (permutation_test_score, StratifiedKFold)
from sklearn.externals import joblib

from my_settings import (data_path, source_folder, step_size, window_size)

# make time points
times = np.arange(-4000, 1001, 1)
times = times / 1000.
selected_times = times[::step_size]
n_time = sum((selected_times + window_size) < times[-1])

# Load data
subjects = [
    "0008", "0009", "0010", "0012", "0014", "0015", "0016", "0017", "0018",
    "0019", "0020", "0021", "0022"
]

cls_all = []
pln_all = []

for subject in subjects:
    cls = np.load(source_folder + "graph_data/%s_cls_pow_sliding.npy" %
                  subject).item()

    pln = np.load(source_folder + "graph_data/%s_pln_pow_sliding.npy" %
                  subject).item()

    cls_tmp = []
    cls_tmp.append(cls["ge_alpha"])
    cls_tmp.append(cls["ge_beta"])
    cls_tmp.append(cls["ge_gamma_low"])
    cls_tmp.append(cls["ge_gamma_high"])

    pln_tmp = []
    pln_tmp.append(pln["ge_alpha"])
    pln_tmp.append(pln["ge_beta"])
    pln_tmp.append(pln["ge_gamma_low"])
    pln_tmp.append(pln["ge_gamma_high"])

    cls_all.append(np.asarray(cls_tmp))
    pln_all.append(np.asarray(pln_tmp))

data_cls = np.asarray(cls_all)
data_pln = np.asarray(pln_all)

# Load GAT model
gat = joblib.load(data_path + "decode_time_gen/gat_ge.jl")

# Setup data for epochs and cross validation
X = np.vstack([data_cls, data_pln])
y = np.concatenate([np.zeros(len(data_cls)), np.ones(len(data_pln))])
cv = StratifiedKFold(n_splits=7, shuffle=True)

perm_score_results = []
for j, est in enumerate(gat.estimators_):
    for tmp in est:
        lr_mean = LogisticRegression(C=0.0001)
        lr_mean.coef_ = np.asarray([lr.coef_ for lr in est]).mean(
            axis=0).squeeze()
        lr_mean.intercept_ = np.asarray([lr.intercept_ for lr in est]).mean()

    score, perm_score, pval = permutation_test_score(
        lr_mean, X[:, :, j], y, cv=cv, scoring="roc_auc", n_permutations=2000)
    perm_score_results.append({
        "score": score,
        "perm_score": perm_score,
        "pval": pval
    })

joblib.dump(perm_score_results,
            data_path + "decode_time_gen/perm_score_results_ge.npy")
