import numpy as np
import bct
from sklearn.externals import joblib
from my_settings import (bands, source_folder)

from sklearn.model_selection import (StratifiedKFold, cross_val_score)
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import AdaBoostClassifier

subjects = [
    "0008", "0009", "0010", "0012", "0014", "0015", "0016", "0017", "0018",
    "0019", "0020", "0021", "0022"
]

cls_all = []
pln_all = []

for subject in subjects:
    cls = np.load(source_folder + "graph_data/%s_cls_pow_sliding.npy" %
                  subject).item()

    pln = np.load(source_folder + "graph_data/%s_pln_pow_sliding.npy" %
                  subject).item()

    cls_tmp = []
    cls_tmp.append(cls["trans_alpha"])
    cls_tmp.append(cls["trans_beta"])
    cls_tmp.append(cls["trans_gamma_low"])
    cls_tmp.append(cls["trans_gamma_high"])

    pln_tmp = []
    pln_tmp.append(pln["trans_alpha"])
    pln_tmp.append(pln["trans_beta"])
    pln_tmp.append(pln["trans_gamma_low"])
    pln_tmp.append(pln["trans_gamma_high"])

    cls_all.append(np.asarray(cls_tmp).reshape(-1))
    pln_all.append(np.asarray(pln_tmp).reshape(-1))

data_cls = np.asarray(cls_all)
data_pln = np.asarray(pln_all)

X = np.vstack([data_cls, data_pln])
y = np.concatenate([np.zeros(len(data_cls)), np.ones(len(data_pln))])

cv = StratifiedKFold(n_splits=7, shuffle=True)

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

scores = cross_val_score(ada_cv, X, y, cv=cv)


# Logistic Regression with cross validation for C
scores = []
coefs = []
Cs = []
LRs = []

for train, test in cv.split(X_slc, y):
    clf = LogisticRegression(C=1)
    #clf = LogisticRegressionCV()
    clf.fit(X[train], y[train])
    y_pred = clf.predict(X[test])

    scores.append(roc_auc_score(y[test], y_pred))
    coefs.append(clf.coef_)
    # Cs.append(clf.C_)
    LRs.append(clf)

lr_mean = LogisticRegression()
lr_mean.coef_ = np.asarray(coefs).mean(axis=0)
lr_mean.C = np.asarray(Cs).mean()
lr_mean.intercept_ = np.asarray([est.intercept_ for est in LRs]).mean()

lr_coef_mean = np.asarray(coefs).mean(axis=0)
lr_coef_std = np.asarray(coefs).std(axis=0)

cv_scores = cross_val_score(
    lr_mean, X_slc, y, scoring="roc_auc", cv=cv)

score_full_X, perm_scores_full_X, pvalue_full_X = permutation_test_score(
    lr_mean,
    X,
    y,
    scoring="roc_auc",
    cv=StratifiedKFold(9),
    n_permutations=2000,
    n_jobs=1, verbose=2)

# RandomizedLogisticRegression feature selection
for i in range(20):
    rlr = RandomizedLogisticRegression(
        n_resampling=5000, C=lr_mean.C, selection_threshold=0.25)
    rlr.fit(X, y)
    print(sum(rlr.get_support()))
    
    
X_rlr = rlr.transform(X)

cv_scores_rlr = cross_val_score(lr_mean, X_rlr, y, 
                                scoring="roc_auc", cv=StratifiedKFold(9))

score_rlr, perm_scores_rlr, pvalue_rlr = permutation_test_score(
    lr_mean,
    X_rlr,
    y,
    scoring="roc_auc",
    cv=StratifiedKFold(9),
    n_permutations=2000,
    n_jobs=2)

