import sys

import h5io
import pandas as pd
from sklearn.externals import joblib
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from skopt import BayesSearchCV
from skopt.space import Integer

from my_settings import erf_mvpa

n_jobs = int(sys.argv[1])

windows_size = 10
tol = 1e-5

seed = 352341561
seed_cv = 23426144

Xy = h5io.read_hdf5(erf_mvpa + "Xy_cls_v_pln_erf_RM.hd5")
X = Xy['X'][:, :, windows_size:-windows_size]
y = Xy['y']

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

pipe = make_pipeline(
    StandardScaler(),  # z-score normalization
    SelectKBest(k=100, score_func=mutual_info_classif),
    LogisticRegression(C=1, tol=tol, solver='lbfgs'))

search_space = {
    'selectkbest__k': Integer(1200, 2000),
}

opt = BayesSearchCV(
    pipe,
    search_space,
    cv=cv,
    n_jobs=n_jobs,
    verbose=1,
    n_iter=50,
    scoring='roc_auc',
    random_state=seed_cv)

results = pd.DataFrame()
opts = []
for ii in (range(Xy['X'].shape[-1])):
    if ii > 200:  # Skip baseline
        print('working on time %s' % (ii))
        X = Xy["X"][:, :, ii]

        opt.fit(X, y)
        opts.append(opt)

        # Generate a row of results
        row = pd.DataFrame([{
            'kbest': opt.best_params_['selectkbest__k'],
            'C': opt.best_params_['logisticregression__C'],
            'best_score': opt.best_score_,
            'time': ii
        }])
        results = pd.concat((results, row))

        results.to_csv(erf_mvpa + 'skopt_results_lr_kbest_C.csv', index=False)

joblib.dump(opts, erf_mvpa + 'skopt_results_lr_opts.pkl')
