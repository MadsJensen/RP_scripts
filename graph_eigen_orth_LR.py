import numpy as np
import bct
from sklearn.externals import joblib
from my_settings import (source_folder)

from sklearn.model_selection import (StratifiedKFold, cross_val_score,
                                     permutation_test_score)
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.metrics import roc_auc_score

subjects = [
    "0008", "0009", "0010", "0012", "0013", "0014", "0015", "0016",
    "0019", "0020", "0021", "0022"
]

tois = ["pln", "pre-press", "post-press"]
cls_all = []
pln_all = []

scores_all = dict()
scores_perm = dict()

for toi in tois:
    cls_all = []
    pln_all = []
    for subject in subjects:
        cls = np.load(source_folder + "graph_data/%s_classic_corr_%s_orth.npy" %
                      (subject, toi))

        pln = np.load(source_folder + "graph_data/%s_plan_corr_%s_orth.npy" %
                      (subject, toi))

        cls_all.append(cls.mean(axis=0))
        pln_all.append(pln.mean(axis=0))

        data_cls = np.asarray([bct.eigenvector_centrality_und(g)
                               for g in cls_all])
        data_pln = np.asarray([bct.eigenvector_centrality_und(g)
                               for g in pln_all])

    X = np.vstack([data_cls, data_pln])
    y = np.concatenate([np.zeros(len(data_cls)), np.ones(len(data_pln))])

    cv = StratifiedKFold(n_splits=6, shuffle=True)
    # Logistic Regression with cross validation for C
    scores = []
    coefs = []
    Cs = []
    LRs = []

    for train, test in cv.split(X, y):
        # clf = LogisticRegression(C=1)
        clf = LogisticRegressionCV()
        clf.fit(X[train], y[train])
        y_pred = clf.predict(X[test])

        scores.append(roc_auc_score(y[test], y_pred))
        coefs.append(clf.coef_)
        Cs.append(clf.C_)
        LRs.append(clf)

    lr_mean = LogisticRegression()
    lr_mean.coef_ = np.asarray(coefs).mean(axis=0)
    lr_mean.C = np.asarray(Cs).mean()
    lr_mean.intercept_ = np.asarray([est.intercept_ for est in LRs]).mean()

    lr_coef_mean = np.asarray(coefs).mean(axis=0)
    lr_coef_std = np.asarray(coefs).std(axis=0)

    scores_all[toi] = cross_val_score(
        lr_mean, X, y, scoring="roc_auc", cv=StratifiedKFold(n_splits=6,
                                                             shuffle=True))

    score_full_X, perm_scores_full_X, pvalue_full_X = permutation_test_score(
        lr_mean,
        X,
        y,
        scoring="roc_auc",
        cv=StratifiedKFold(n_splits=6, shuffle=True),
        n_permutations=2000,
        n_jobs=1)

    scores_perm[toi] = [score_full_X, perm_scores_full_X, pvalue_full_X]

    # save the classifier
    joblib.dump(
        lr_mean,
        source_folder + "graph_data/sk_models/eigen_LR_%s_orth.plk" %
        toi)

np.save(source_folder + "graph_data/eigen_scores_all_LR.npy",
        scores_all)
