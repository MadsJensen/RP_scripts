import sys
import numpy as np
import h5io
from mne.decoding import (SlidingEstimator, cross_val_multiscore, LinearModel)
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from my_settings import beamformer_mvpa, bands

n_jobs = int(sys.argv[1])
seed = 234351
tol = 1e-5
window_size = 10

for band in bands:
    Xy = h5io.read_hdf5(beamformer_mvpa + "Xy_cls_v_pln_%s_RM.hd5" % band)
    X = Xy['X'][:, :, window_size:-window_size]
    y = Xy['y']

    groups_cv = np.repeat(np.arange(0, (len(y) / 2), 1), 2)
    cv = LeaveOneGroupOut()

    clf = make_pipeline(
        StandardScaler(),  # z-score normalization
        LinearModel(LogisticRegression(C=1, tol=tol, solver='lbfgs')))

    time_decod = SlidingEstimator(clf, n_jobs=n_jobs, scoring='roc_auc')

    time_decod.fit(X, y)
    joblib.dump(
        time_decod,
        beamformer_mvpa + "source_cls_v_pln_itc_logreg_%s_RM.jbl" % band)

    scores = cross_val_multiscore(time_decod, X, y, cv=cv, groups=groups_cv)
    h5io.write_hdf5(
        beamformer_mvpa + "source_cls_v_pln_itc_logreg_%s_RM.hd5" % band,
        scores)
