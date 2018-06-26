import sys
import numpy as np
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from mne.decoding import (SlidingEstimator, cross_val_multiscore, LinearModel)

from my_settings import erf_mvpa

seed = 23462146345329

condition_0 = sys.argv[1]
condition_1 = sys.argv[2]
n_jobs = int(sys.argv[3])

X_0 = np.load(erf_mvpa + "X_%s_erf.npy" % condition_0)
X_1 = np.load(erf_mvpa + "X_%s_erf.npy" % condition_1)

X = np.concatenate((X_0, X_1), axis=0)
y = np.concatenate((np.zeros(len(X_0)), np.ones(len(X_1))))

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

clf = make_pipeline(
    StandardScaler(),  # z-score normalization
    LinearModel(LogisticRegression(C=1)))
time_decod = SlidingEstimator(clf, n_jobs=n_jobs, scoring='roc_auc')

time_decod.fit(X, y)
joblib.dump(
    time_decod, erf_mvpa +
    "source_%s_v_%s_itc_evk_logreg_erf.jbl" % (condition_0, condition_1))

scores = cross_val_multiscore(time_decod, X, y, cv=cv)
np.save(
    erf_mvpa +
    "source_%s_v_%s_itc_evk_logreg_erf_FS.npy" % (condition_0, condition_1),
    scores)
