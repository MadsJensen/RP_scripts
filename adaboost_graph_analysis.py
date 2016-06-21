import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.cross_validation import (StratifiedKFold, LeaveOneOut,
                                      cross_val_score)
from sklearn.grid_search import GridSearchCV
# from sklearn.pipeline import make_pipeline

from my_settings import *

subjects = ["0008", "0009", "0010", "0012", "0013", "0014", "0015", "0016",
            "0017", "0018", "0019", "0020", "0021", "0022"]

for subject in subjects:
    cls = np.load(source_folder + "graph_data/%s_classic_pow_pln.npy" %
                  subject)
    pln = np.load(source_folder + "graph_data/%s_plan_pow_pln.npy" % subject)

results_all = {}

for band in bands:
    results_cls = []
    results_pln = []
    results_cls.append(cls[band].mean(axis=0))
    results_pln.append(pln[band].mean(axis=0))

    idx = np.tril_indices_from(results_cls[0], k=-1)

    data_cls = []
    data_pln = []

    for j in range(len(results_cls)):
        data_cls += [results_cls[j][idx]]
        data_pln += [results_pln[j][idx]]

    data_cls = np.asarray(data_cls)
    data_pln = np.asarray(data_pln)

    X = np.vstack([data_cls, data_pln])
    y = np.concatenate([np.zeros(len(data_cls)), np.ones(len(data_pln))])

    cv = StratifiedKFold(y, n_folds=10)
    llo = LeaveOneOut(len(y))

    ada = AdaBoostClassifier()

    adaboost_params = {"n_estimators": np.arange(20, 500, 20),
                       "learning_rate": np.arange(0.1, 1.1, 0.1)}

    grid = GridSearchCV(ada,
                        param_grid=adaboost_params,
                        cv=cv,
                        verbose=2,
                        n_jobs=6)
    grid.fit(X2, y)

    ada_cv = grid.best_estimator_

    scores = cross_val_score(ada_cv, X2, y, cv=cv, scoring="roc_auc")

    results_all["%s_scores" % band] = scores
    results_all["%s_best_est" % band] = ada_cv
