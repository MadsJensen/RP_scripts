import bct
import numpy as np
import pandas as pd
from sklearn.externals import joblib
from sklearn.linear_model import (LogisticRegression, LogisticRegressionCV,
                                  RandomizedLogisticRegression)
from sklearn.metrics import accuracy_score
from sklearn.model_selection import (StratifiedKFold, cross_val_score,
                                     permutation_test_score)

from my_settings import (source_folder, results_folder)

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

        scores.append(accuracy_score(y[test], y_pred))
        coefs.append(clf.coef_)
        Cs.append(clf.C_)
        LRs.append(clf)

    # Make LR model from mean of different models
    lr_mean = LogisticRegression()
    lr_mean.coef_ = np.asarray(coefs).mean(axis=0)
    lr_mean.C = np.asarray(Cs).mean()
    lr_mean.intercept_ = np.asarray([est.intercept_ for est in LRs]).mean()

    lr_coef_mean = np.asarray(coefs).mean(axis=0)
    lr_coef_std = np.asarray(coefs).std(axis=0)

    # Calc roc auc score from mean model
    scores_all[toi] = cross_val_score(
        lr_mean, X, y, scoring="accuracy", cv=StratifiedKFold(n_splits=6,
                                                              shuffle=True))

    # Test model significiant of mean model w/ permutation test
    score_full_X, perm_scores_full_X, pvalue_full_X = permutation_test_score(
        lr_mean,
        X,
        y,
        scoring="accuracy",
        cv=StratifiedKFold(n_splits=6, shuffle=True),
        n_permutations=2000,
        n_jobs=1)

    # Save permutation scores
    scores_perm[toi] = [score_full_X, perm_scores_full_X, pvalue_full_X]

    # RandomizedLogisticRegression feature selection
    # Grid search for optimal RLR params
    selection_threshold = np.arange(0.1, 1, 0.05)
    rlr_grid_search = pd.DataFrame()

    for st in selection_threshold:
        for i in range(10):
            print("Working on: %s (%d of 200)" % (st, (i + 1)))
            rlr = RandomizedLogisticRegression(
                n_resampling=200,
                C=lr_mean.C,
                sample_fraction=0.8,
                selection_threshold=st,
                n_jobs=1)
            rlr.fit(X, y)
            X_rlr = rlr.transform(X)

            if X_rlr.size:
                cv_scores_rlr = cross_val_score(
                    lr_mean, X_rlr, y, scoring="accuracy", cv=cv)

                rlr_tmp = {
                    "st": st,
                    "cv_score": cv_scores_rlr.mean(),
                    "cv_std": cv_scores_rlr.std(),
                    "n_features": sum(rlr.get_support())
                }
                rlr_grid_search = rlr_grid_search.append(
                    rlr_tmp, ignore_index=True)

    rlr_grid_search_mean = rlr_grid_search.groupby(by="st").mean()
    rlr_grid_search_mean["n_feat_std"] = \
        rlr_grid_search.groupby(by="st").std()["n_features"]
    rlr_grid_search_mean["cv_score_std"] = rlr_grid_search.groupby(
        by="st").std()["cv_score"]

    # take st param from grid search
    st = rlr_grid_search_mean.st[rlr_grid_search_mean.cv_score.argmax()]
    rlr = RandomizedLogisticRegression(
        n_resampling=5000, C=lr_mean.C, selection_threshold=st)
    rlr.fit(X, y)
    X_rlr = rlr.transform(X)

    cv_scores_rlr = cross_val_score(
        lr_mean, X_rlr, y, scoring="accuracy",
        cv=StratifiedKFold(6, shuffle=True))

    score_rlr, perm_scores_rlr, pvalue_rlr = permutation_test_score(
        lr_mean,
        X_rlr,
        y,
        scoring="accuracy",
        cv=StratifiedKFold(6, shuffle=True),
        n_permutations=2000,
        n_jobs=2)

    # save the classifier
    joblib.dump(
        lr_mean,
        source_folder + "graph_data/sk_models/eigen_LR_%s_orth.plk" %
        toi)
    # save rlr classifier
    joblib.dump(
        rlr,
        source_folder + "graph_data/sk_models/eigen_LR_%s_orth_RLR.plk" %
        toi)
    rlr_grid_search_mean.to_csv(
        results_folder + "graph_data/eigen_LR_%s_orth_RLR_scores.csv" %
        toi)

np.save(results_folder + "graph_data/eigen_scores_all_LR.npy",
        scores_all)
np.save(results_folder + "graph_data/eigen_perm_scores_all_LR.npy",
        scores_perm)
