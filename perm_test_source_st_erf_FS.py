# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 17:06:22 2017

@author: mje
"""
import numpy as np
from sklearn.externals import joblib
from sklearn.model_selection import StratifiedKFold, permutation_test_score
from tqdm import tqdm
import pandas as pd

from slack_test import send_slack

from my_settings import erf_mvpa

# Make dataframe with threshold
rocs = np.load(erf_mvpa + "source_cls_v_pln_itc_evk_logreg_erf_FS.npy",
               ).mean(axis=0)
times = np.arange(-3500, 501, 2) * 1e-3

df_roc = pd.DataFrame(data=rocs, index=times, columns=["erf"])
threshold = df_roc[df_roc.index < 0].mean() + 2 * df_roc[df_roc.index < 0].std(
)
df_threshold = df_roc > threshold

ests = joblib.load(erf_mvpa + "source_cls_v_pln_itc_evk_logreg_erf_FS.jbl")

X = np.load(erf_mvpa + "X_cls_v_pln_erf.npy")
y = np.load(erf_mvpa + "y_cls_v_pln_erf.npy")

cv = StratifiedKFold(n_splits=5, shuffle=True)

perm_res = {}
send_slack("RP_int: ERF permutation starting")
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

np.save(erf_mvpa + "perm_source_st_cls_v_pln_itc_evk_logreg_erf.npy", perm_res)

send_slack("RP_int: ERF permutation done")
