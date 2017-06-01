import numpy as np
import bct
from sklearn.externals import joblib
from my_settings import *
from sklearn.model_selection import (StratifiedKFold, cross_val_score,
                                     GridSearchCV)
from sklearn.ensemble import AdaBoostClassifier

subjects = [
    "0008", "0009", "0010", "0012", "0013", "0014", "0015", "0016",
    "0019", "0020", "0021", "0022"
]

cls_all = []
pln_all = []

tois = ["pln", "pre-press", "post-press"]
scores_all = dict()

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

    cls_all_2 = np.asarray(cls_all)
    pln_all_2 = np.asarray(pln_all)

    full_matrix = np.concatenate([cls_all_2, pln_all_2], axis=0)

    threshold = np.median(full_matrix[np.nonzero(full_matrix)]) + \
        np.std(full_matrix[np.nonzero(full_matrix)])

    data_cls_bin = cls_all > threshold
    data_pln_bin = pln_all > threshold

    data_cls = [np.asarray([bct.strengths_und(g)
                            for g in data_cls_bin])]
    data_pln = [np.asarray([bct.strengths_und(g)
                            for g in data_pln_bin])]

    X = np.vstack([data_cls, data_pln])
    y = np.concatenate([np.zeros(len(data_cls)), np.ones(len(data_pln))])

    cv = StratifiedKFold(n_splits=6, shuffle=True)

    cv_params = {"learning_rate": np.arange(0.1, 1.1, 0.1),
                 'n_estimators': np.arange(1, 80, 2)}

    grid = GridSearchCV(AdaBoostClassifier(),
                        cv_params,
                        scoring='roc_auc',
                        cv=cv,
                        n_jobs=1,
                        verbose=2)
    grid.fit(X, y)
    ada_cv = grid.best_estimator_

    scores = cross_val_score(ada_cv, X, y, cv=cv, scoring="roc_auc")
    scores_all[toi] = scores

    # save the classifier
    joblib.dump(
        ada_cv,
        source_folder + "graph_data/sk_models/path-strength_ada_%s_pln.plk" %
        toi)

np.save(source_folder + "graph_data/path-strength_scores_all_pln_orth.npy",
        scores_all)
