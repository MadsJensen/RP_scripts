import numpy as np
import bct
from sklearn.externals import joblib
from scipy.io import loadmat
from my_settings import (source_folder)

from sklearn.ensemble import AdaBoostClassifier
from sklearn.cross_validation import (StratifiedKFold, cross_val_score)
from sklearn.grid_search import GridSearchCV

subjects = [
    "0008", "0009", "0010", "0012", "0013", "0014", "0015", "0016", "0018",
    "0019", "0020", "0022"
]

cls_all = []
pln_all = []

for subject in subjects:
    cls = loadmat(
        source_folder +
        "ave_ts/mat_files/%s_classic_ts_DKT_snr-3-epo.mat" % subject)["data"]

    pln = loadmat(source_folder +
                  "ave_ts/mat_files/%s_plan_ts_DKT_snr-3-epo.mat" % subject)[
                      "data"]

    cls_all.append(cls)
    pln_all.append(pln)

data_cls = []
for j in range(len(cls_all)):
    data_cls.append(
        np.asarray([bct.charpath(g) for g in cls_all]).mean(axis=0))
data_pln = []
for j in range(len(pln_all)):
    data_pln.append(np.asarray([bct.charpath(g) for g in pln]).mean(axis=0))

data_cls = np.asarray(data_cls)
data_pln = np.asarray(data_pln)

X = np.vstack([data_cls, data_pln])
y = np.concatenate([np.zeros(len(data_cls)), np.ones(len(data_pln))])

cv = StratifiedKFold(y, n_folds=6, shuffle=True)

cv_params = {
    "learning_rate": np.arange(0.1, 1.1, 0.1),
    'n_estimators': np.arange(1, 80, 2)
}

grid = GridSearchCV(
    AdaBoostClassifier(),
    cv_params,
    scoring='roc_auc',
    cv=cv,
    n_jobs=6,
    verbose=1)
grid.fit(X, y)
ada_cv = grid.best_estimator_

scores = cross_val_score(ada_cv, X, y, cv=cv)

# save the classifier
joblib.dump(ada_cv,
            source_folder + "graph_data/sk_models/char-path_ada_pln_orth.plk")

np.save(source_folder + "graph_data/char-path_scores_all_pln_orth.npy",
        scores)
