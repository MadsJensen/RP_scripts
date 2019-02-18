import sys
import h5io
import numpy as np
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectPercentile, f_classif

from mne.decoding import (SlidingEstimator, cross_val_multiscore, LinearModel)

from my_settings import erf_mvpa

n_jobs = int(sys.argv[1])

seed = 352341561

Xy = h5io.read_hdf5(erf_mvpa + "X_cls_v_pln_erf_RM.npy")
X = Xy['X']
y = Xy['y']

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

clf = make_pipeline(
    StandardScaler(),  # z-score normalization
    LinearModel(LogisticRegression(C=1)))
time_decod = SlidingEstimator(clf, n_jobs=n_jobs, scoring='roc_auc')

time_decod.fit(X, y)
joblib.dump(time_decod,
            erf_mvpa + "source_cls_v_pln_evk_logreg_erf_full.jbl")

scores = cross_val_multiscore(time_decod, X, y, cv=cv)
h5io.write_hdf5(erf_mvpa + "source_cls_v_pln_evk_logreg_erf_full.npy",
                scores)
