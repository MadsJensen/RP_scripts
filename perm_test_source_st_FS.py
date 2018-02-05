# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 17:06:22 2017

@author: mje
"""
import sys

import numpy as np
from sklearn.externals import joblib
from sklearn.model_selection import StratifiedKFold, permutation_test_score
from tqdm import tqdm
import pandas as pd

from my_settings import beamformer_mvpa

band = sys.argv[1]

# Make dataframe with threshold
rocs = np.load(beamformer_mvpa + "source_cls_v_pln_itc_evk_logreg_%s_FS.npy" %
               (band)).mean(axis=0)
times = np.arange(-3500, 501, 2) * 1e-3

df_roc = pd.DataFrame(data=rocs.T, index=times, columns=[band])
threshold = df_roc[df_roc.index < 0].mean() + 2 * df_roc[df_roc.index < 0].std(
)
df_threshold = df_roc > threshold

ests = joblib.load(
    beamformer_mvpa + "source_cls_v_pln_itc_evk_logreg_%s_FS.jbl" % (band))

X = np.load(beamformer_mvpa + "X_cls_v_pln_%s.npy" % (band))
y = np.load(beamformer_mvpa + "y_cls_v_pln.npy")

cv = StratifiedKFold(n_splits=5, shuffle=True)

perm_res = {}
for jj, status in tqdm(enumerate(df_threshold.values)):
    if status:
        X_test = X[:, :, jj]
        perm_res["%s" % jj] = permutation_test_score(
            ests.estimators_[jj],
            X_test,
            y,
            cv=cv,
            n_permutations=2000,
            scoring="roc_auc",
            n_jobs=1)

np.save(beamformer_mvpa + "perm_source_st_cls_v_pln_itc_evk_logreg_%s.npy" %
        (band), perm_res)
