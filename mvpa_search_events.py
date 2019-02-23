import sys

import h5io
import mne
import numpy as np
from mne.decoding import get_coef
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

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

    models = joblib.load(beamformer_mvpa +
                         "source_cls_v_pln_itc_evk_logreg_%s_RM.jbl" % band)

    scores = h5io.read_hdf5(beamformer_mvpa +
                            "source_cls_v_pln_itc_evk_logreg_%s_RM.hd5" % band)

    time_idx_start = stc.time_as_index(-2.50)[0]
    time_idx_end = stc.time_as_index(-2.20)[0]

    coefs = get_coef(models)

    mean_coefs = coefs[:, time_idx_start:time_idx_end].mean(axis=-1)

    mean_model = models.estimators_[time_idx_start].named_steps[
        'linearmodel'].model
    mean_model.coefs_ = mean_coefs

    rocs = np.zeros(len(models.estimators_))
    for ii in range(len(models.estimators_)):
        X_std = models.estimators_[
            time_idx_start + 750].named_steps['standardscaler'].transform(
                X[:, :, ii])
        rocs[ii] = accuracy_score(y, mean_model.predict(X_std))
        h5io.write_hdf5(
            beamformer_mvpa + 'mean_model_rocs_%s_acc.hd5' % band,
            rocs,
            overwrite=True)
