import sys
import numpy as np
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from mne.decoding import (SlidingEstimator, cross_val_multiscore, LinearModel)
import h5io

from my_settings import erf_mvpa

seed = 2346219634
tol = 1e-5

condition_0 = sys.argv[1]
condition_1 = sys.argv[2]
n_jobs = int(sys.argv[3])

X_0 = np.load(erf_mvpa + "X_%s_erf_RM.npy" % condition_0)
X_1 = np.load(erf_mvpa + "X_%s_erf_RM.npy" % condition_1)

X_0 = np.delete(X_0, 10, 0)
X_1 = np.delete(X_1, 10, 0)

X = np.concatenate((X_0, X_1), axis=0)
y = np.concatenate((np.zeros(len(X_0)), np.ones(len(X_1))))

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

clf = make_pipeline(
    StandardScaler(),  # z-score normalization
    SelectFromModel(
        LogisticRegression(C=1, solver='lbfgs', tol=tol),
        threshold="1.5*mean"),
    LinearModel(LogisticRegression(C=1, solver='lbfgs', tol=tol)))
time_decod = SlidingEstimator(clf, n_jobs=n_jobs, scoring='roc_auc')

time_decod.fit(X, y)
joblib.dump(
    time_decod, erf_mvpa +
    "st_%s_v_%s_evk_logreg_erf_RM_sfm_2.jbl" % (condition_0, condition_1))

scores = cross_val_multiscore(time_decod, X, y, cv=cv)
h5io.write_hdf5(
    erf_mvpa +
    "st_%s_v_%s_evk_logreg_erf_RM_sfm_2.hd5" % (condition_0, condition_1),
    scores,
    overwrite=True)
