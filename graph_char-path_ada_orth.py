import bct
import numpy as np
from scipy.io import loadmat
from sklearn.ensemble import AdaBoostClassifier
from sklearn.externals import joblib
from sklearn.model_selection import (GridSearchCV, StratifiedKFold,
                                     cross_val_score)

from my_settings import (source_folder)

subjects = [
    "0008", "0009", "0010", "0012", "0013", "0014", "0015", "0016", "0018",
    "0019", "0020", "0022"
]

cls_all = []
pln_all = []
for subject in subjects:
    cls = np.load(source_folder + "graph_data/%s_classic_corr_pln_orth.npy" %
                  subject)

    pln = np.load(source_folder + "graph_data/%s_plan_corr_pln_orth.npy"  %
                  subject)

    cls_all.append(cls)
    pln_all.append(pln)

data_cls = np.asarray([bct.charpath(g) for g in cls_all]).mean(axis=0)
data_pln = np.asarray([bct.charpath(g) for g in pln_all]).mean(axis=0)

data_cls = np.asarray(data_cls)
data_pln = np.asarray(data_pln)

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

scores = cross_val_score(ada_cv, X, y, cv=cv, scoring="roc_auc")

# save the classifier
joblib.dump(ada_cv,
            source_folder + "graph_data/sk_models/char-path_ada_pln_orth.plk")

np.save(source_folder + "graph_data/char-path_scores_all_pln_orth.npy",
        scores)
