import numpy as np
import bct
from sklearn.externals import joblib
from my_settings import *

from sklearn.ensemble import AdaBoostClassifier
from sklearn.cross_validation import (StratifiedShuffleSplit, cross_val_score)
from sklearn.grid_search import GridSearchCV

subjects = ["0008", "0009", "0010", "0012", "0014", "0015", "0016", "0017",
            "0018", "0019", "0020", "0021", "0022"]

cls_all = []
pln_all = []

scores_all = np.empty([4, 10])

for subject in subjects:
    cls = np.load(source_folder + "graph_data/%s_classic_pow_post.npy" %
                  subject).item()

    pln = np.load(source_folder + "graph_data/%s_plan_pow_post.npy" %
                  subject).item()

    cls_all.append(cls)
    pln_all.append(pln)

for k, band in enumerate(bands.keys()):
    data_cls = []
    for j in range(len(cls_all)):
        tmp = cls_all[j][band]
        data_cls.append(np.asarray([bct.efficiency_wei(g)
                                    for g in tmp]).mean(axis=0))
    data_pln = []
    for j in range(len(pln_all)):
        tmp = pln_all[j][band]
        data_pln.append(np.asarray([bct.efficiency_wei(g)
                                    for g in tmp]).mean(axis=0))

    data_cls = np.asarray(data_cls)
    data_pln = np.asarray(data_pln)

    X = np.vstack([data_cls, data_pln])
    y = np.concatenate([np.zeros(len(data_cls)), np.ones(len(data_pln))])

    cv = StratifiedShuffleSplit(y, test_size=0.1)

    cv_params = {"learning_rate": np.arange(0.1, 1.1, 0.1),
                 'n_estimators': np.arange(1, 80, 2)}

    grid = GridSearchCV(AdaBoostClassifier(),
                        cv_params,
                        scoring='accuracy',
                        cv=cv,
                        n_jobs=4,
                        verbose=1)
    grid.fit(X, y)
    ada_cv = grid.best_estimator_

    scores = cross_val_score(ada_cv, X, y, cv=cv)
    scores_all[k, :] = scores

    # save the classifier
    joblib.dump(
        ada_cv,
        source_folder + "graph_data/transitivity_ada_post_%s.plk" % band)

np.save(source_folder + "graph_data/transitivity_scores_all_post.npy",
        scores_all)
