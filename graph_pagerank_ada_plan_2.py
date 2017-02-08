import numpy as np
import bct
from sklearn.externals import joblib
from my_settings import (bands, source_folder)

from sklearn.ensemble import AdaBoostClassifier
from sklearn.cross_validation import (StratifiedKFold, cross_val_score)
from sklearn.grid_search import GridSearchCV

subjects = [
    "0008", "0009", "0010", "0012", "0014", "0015", "0016", "0017", "0018",
    "0019", "0020", "0021", "0022"
]

cls_all = []
pln_all = []

scores_all = np.empty([4, 7])

for subject in subjects:
    cls = np.load("%s_classic_pow_plan.npy" % subject).item()

    pln = np.load("%s_plan_pow_plan.npy" % subject).item()

    cls_all.append(cls)
    pln_all.append(pln)

for k, band in enumerate(bands.keys()):
    data_cls = []
    for j in range(len(cls_all)):
        tmp = cls_all[j][band]
        data_cls.append(
            np.asarray(
                [bct.centrality.pagerank_centrality(
                    g, d=0.85) for g in tmp]).mean(axis=0))
    data_pln = []
    for j in range(len(pln_all)):
        tmp = pln_all[j][band]
        data_pln.append(
            np.asarray(
                [bct.centrality.pagerank_centrality(
                    g, d=0.85) for g in tmp]).mean(axis=0))

    data_cls = np.asarray(data_cls)
    data_pln = np.asarray(data_pln)

    X = np.vstack([data_cls, data_pln])
    y = np.concatenate([np.zeros(len(data_cls)), np.ones(len(data_pln))])

    cv = StratifiedKFold(y, n_folds=7, shuffle=True)

    cv_params = {
        "learning_rate": np.arange(0.1, 1.1, 0.1),
        'n_estimators': np.arange(1, 80, 2)
    }

    grid = GridSearchCV(
        AdaBoostClassifier(),
        cv_params,
        scoring='accuracy',
        cv=cv,
        n_jobs=1,
        verbose=1)
    grid.fit(X, y)
    ada_cv = grid.best_estimator_

    scores = cross_val_score(ada_cv, X, y,
                             cv=StratifiedKFold(y, n_folds=7, shuffle=True),
                                               scoring="accuracy")
    scores_all[k, :] = scores

    # save the classifier
    joblib.dump(
        ada_cv,
        "sk_models/pagerank_ada_plan_%s_2.plk" % band)

np.save("pagerank_scores_all_plan_2.npy", scores_all)
