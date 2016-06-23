import numpy as np
import bct
from sklearn.externals import joblib
from my_settings import *

from sklearn.cross_validation import (StratifiedShuffleSplit, cross_val_score)
from sklearn.grid_search import GridSearchCV

import xgboost as xgb

subjects = ["0008", "0009", "0010", "0012", "0014", "0015", "0016",
            "0017", "0018", "0019", "0020", "0021", "0022"]

cls_all = []
pln_all = []

scores_all = np.empty([4, 10])

for subject in subjects:
    cls = np.load(source_folder + "graph_data/%s_classic_pow_pln.npy" %
                  subject).item()

    pln = np.load(source_folder + "graph_data/%s_plan_pow_pln.npy" %
                  subject).item()

    cls_all.append(cls)
    pln_all.append(pln)

for k, band in enumerate(bands.keys()):
    data_cls = []
    for j in range(len(cls_all)):
        tmp = cls_all[j][band]
        data_cls.append(np.asarray([bct.eigenvector_centrality_und(g)
                                    for g in tmp]).mean(axis=0))
    data_pln = []
    for j in range(len(pln_all)):
        tmp = pln_all[j][band]
        data_pln.append(np.asarray([bct.eigenvector_centrality_und(g)
                                    for g in tmp]).mean(axis=0))

    data_cls = np.asarray(data_cls)
    data_pln = np.asarray(data_pln)

    X = np.vstack([data_cls, data_pln])
    y = np.concatenate([np.zeros(len(data_cls)), np.ones(len(data_pln))])

    cv = StratifiedShuffleSplit(y, test_size=0.1)

    cv_params = {"learning_rate": np.arange(0.1, 1.1, 0.1),
                 "max_depth": [1,2,3,4,5,6,7]}

    grid = GridSearchCV(xgb.XGBClassifier(n_estimators=500),
                        cv_params,
                        scoring='accuracy',
                        cv=cv,
                        n_jobs=1,
                        verbose=1)
    grid.fit(X, y)
    xgb_cv = grid.best_estimator_

    scores = cross_val_score(xgb_cv, X, y, cv=cv)
    scores_all[k, :] = scores

    # save the classifier
    joblib.dump(
        xgb_cv,
        source_folder + "graph_data/sk_models/eigen_xgb_%s.plk" % band)

np.save(source_folder + "graph_data/eigen_scores_all_xgb.npy", scores_all)
