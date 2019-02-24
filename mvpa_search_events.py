import sys

import h5io
import mne
import numpy as np
from mne.decoding import get_coef
from sklearn.externals import joblib
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from my_settings import beamformer_mvpa, beamformer_results, bands

n_jobs = int(sys.argv[1])
seed = 234351
tol = 1e-5
window_size = 10
stc = mne.read_source_estimate(beamformer_results +
                               '0008_classic_Alpha_cor_avg')
stc.crop(stc.times[10], stc.times[-9])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

for band in bands:
    Xy = h5io.read_hdf5(beamformer_mvpa + "Xy_cls_v_pln_%s_RM.hd5" % band)
    X = Xy['X'][:, :, window_size:-window_size]
    y = Xy['y']

    # TODO Better time index
    time_idx_start = stc.time_as_index(-2.50)[0]
    time_idx_end = stc.time_as_index(-2.20)[0]

    models = []
    std = []
    stdscl = StandardScaler()
    clf = LogisticRegression(tol=tol, C=1, solver='lbfgs')
    scores = []
    for time in range(time_idx_start, time_idx_end, 1):
        for train, test in cv.split(X, y):
            X_train, X_test = X[train, :, time], X[test, :, time]
            y_train, y_test = y[train], y[test]
            stdscl.fit(X_train)
            X_train = stdscl.transform(X_train)
            X_test = stdscl.transform(X_test)
            clf.fit(X_train, y_train)
            scores.append(roc_auc_score(y_test, clf.predict(X_test)))
            models.append(clf)
            std.append(stdscl)

    mean_coefs = np.mean([c.coef_ for c in models], axis=0)
    mean_model = models[0]
    mean_model.coef_ = mean_coefs

    mean_std = std[0]
    mean_std.mean_ = np.mean([c.mean_ for c in std], axis=0)
    mean_std.var_ = np.mean([c.var_ for c in std], axis=0)
    mean_std.scale_ = np.mean([c.scale_ for c in std], axis=0)

    rocs = np.zeros(X.shape[-1])

    for tt in range(X.shape[-1]):
        X_std = mean_std.transform(X[:, :, tt])
        rocs[tt] = roc_auc_score(y, mean_model.predict(X_std))
        h5io.write_hdf5(
            beamformer_mvpa + 'mean_model_rocs_%s_roc.hd5' % band,
            rocs,
            overwrite=True)
