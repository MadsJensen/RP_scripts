import sys
import numpy as np
import h5io
import pandas as pd
from mne.decoding import (SlidingEstimator, cross_val_multiscore, LinearModel)
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.feature_selection import SelectFromModel

from my_settings import beamformer_mvpa, bands

n_jobs = int(sys.argv[1])
seed = 234351
tol = 1e-5
window_size = 10

for band in bands:
    Xy = h5io.read_hdf5(beamformer_mvpa + "Xy_cls_v_pln_%s_RM.hd5" % band)
    X = Xy['X'][:, :, window_size:-window_size]
    y = Xy['y']

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    models = joblib.load(beamformer_mvpa +
                         "source_cls_v_pln_itc_evk_logreg_%s_RM.jbl" % band)

    scores_orig = h5io.read_hdf5(beamformer_mvpa +
                            "source_cls_v_pln_itc_evk_logreg_%s_RM.hd5" % band)

    scores = scores.mean(axis=0)

    max_idx = scores.argmax()

    thresholds = np.arange(0.1, 5.1, 0.1)
    est = LogisticRegression(C=1, tol=tol, solver='lbfgs')

    X_std = X[:, :, max_idx]
    X_std = models.estimators_[max_idx].named_steps[
        'standardscaler'].transform(X_std)

    sfm_results = pd.DataFrame()

    for thres in thresholds:
        sfm = SelectFromModel(
            models.estimators_[max_idx].named_steps['linearmodel'].model,
            prefit=True,
            threshold='{:0.03f} * mean'.format(thres))
        X_sfm = sfm.transform(X_std)
        if sum(sfm.get_support() > 0):
            cv_score = cross_val_score(est, X_sfm, y, cv=cv, scoring='roc_auc')

            row = pd.DataFrame([{
                'threshold': thres,
                'mean_score': cv_score.mean(),
                'std_score': cv_score.std(),
                'n_features': sum(sfm.get_support())
            }])
            sfm_results = pd.concat((sfm_results, row))

    clf = make_pipeline(
        StandardScaler(),  # z-score normalization
        SelectFromModel(
            LogisticRegression(C=1, tol=tol, solver='lbfgs'),
            prefit=False,
            threshold='1.6 * mean'),
        LinearModel(LogisticRegression(C=1, tol=tol, solver='lbfgs')))

    time_decod = SlidingEstimator(clf, n_jobs=n_jobs, scoring='roc_auc')
    time_decod.fit(X, y)
    scores = cross_val_multiscore(time_decod, X, y, cv=cv)

sfm_results.sort_values