import numpy as np
import bct
from sklearn.externals import joblib
from my_settings import (source_folder)

from sklearn.model_selection import (StratifiedKFold, cross_val_score,
                                     GridSearchCV)
from sklearn.ensemble import AdaBoostClassifier

subjects = [
    "0008", "0009", "0010", "0012", "0013", "0014", "0015", "0016",
    "0019", "0020", "0021", "0022"
]

tois = ["pln", "pre-press", "post-press"]
cls_all = []
pln_all = []

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

        data_cls = np.asarray([bct.eigenvector_centrality_und(g)
                               for g in cls_all])
        data_pln = np.asarray([bct.eigenvector_centrality_und(g)
                               for g in pln_all])

    X = np.vstack([data_cls, data_pln])
    y = np.concatenate([np.zeros(len(data_cls)), np.ones(len(data_pln))])

    cv = StratifiedKFold(n_splits=6, shuffle=True)

    cv_params = {
        "learning_rate": np.arange(0.1, 1.1, 0.1),
        'n_estimators': np.arange(1, 80, 2)
    }

    grid = GridSearchCV(
        AdaBoostClassifier(),
        cv_params,
        scoring='roc_auc',
        cv=cv,
        n_jobs=1,
        verbose=1)
    grid.fit(X, y)
    ada_cv = grid.best_estimator_

    scores_all[toi] = cross_val_score(ada_cv, X, y, cv=cv,
                                      scoring="roc_auc")

    # save the classifier
    joblib.dump(
        ada_cv,
        source_folder + "graph_data/sk_models/eigen_ada_%s_orth.plk" %
        toi)

    np.save(source_folder + "graph_data/eigen_scores_all_ada_%s.npy" % toi,
            scores_all)
