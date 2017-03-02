import numpy as np
import bct
from sklearn.externals import joblib
from my_settings import (bands, source_folder)

from sklearn.cross_validation import (StratifiedKFold, cross_val_score)
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import AdaBoostClassifier

subjects = [
    "0008", "0009", "0010", "0012", "0014", "0015", "0016", "0017", "0018",
    "0019", "0020", "0021", "0022"
]

cls_all = []
pln_all = []

for subject in subjects:
    cls = np.load(source_folder + "graph_data/%s_classic_pow_pln.npy" %
                  subject).item()

    pln = np.load(source_folder + "graph_data/%s_plan_pow_pln.npy" %
                  subject).item()

    cls_all.append(cls["trans_alpha"])
    pln_all.append(pln["trans_alpha"])

data_cls = np.asarray(cls_all)
data_pln = np.asarray(pln_all)

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
    scoring='accuracy',
    cv=cv,
    n_jobs=1,
    verbose=1)
grid.fit(X, y)
ada_cv = grid.best_estimator_

scores = cross_val_score(ada_cv, X, y, cv=cv)
