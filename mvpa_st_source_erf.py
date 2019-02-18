import sys
import h5io
import numpy as np
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel

from mne.decoding import (SlidingEstimator, cross_val_multiscore, LinearModel)

from my_settings import erf_mvpa
windows_size = 10

n_jobs = int(sys.argv[1])

seed = 352341561

Xy = h5io.read_hdf5(erf_mvpa + "Xy_cls_v_pln_erf_RM.hd5")
X = Xy['X'][:, :, windows_size:-windows_size]
y = Xy['y']

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

clf = make_pipeline(
    StandardScaler(),  # z-score normalization
    SelectFromModel(LinearSVC(penalty="l1")),
    LinearModel(LogisticRegression(C=1)))
time_decod = SlidingEstimator(clf, n_jobs=n_jobs, scoring='roc_auc')

time_decod.fit(X, y)
joblib.dump(time_decod, erf_mvpa + "source_cls_v_pln_evk_logreg_erf_sfm.jbl")

scores = cross_val_multiscore(time_decod, X, y, cv=cv)
h5io.write_hdf5(erf_mvpa + "source_cls_v_pln_evk_logreg_erf_sfm.hd5", scores)
