import numpy as np
import mne
from my_settings import (beamformer_results, beamformer_mvpa, bands, subjects)

for band in bands[:1]:
    for j, subject in enumerate(subjects):
        stc_cls = mne.read_source_estimate(
            beamformer_results + "%s_classic_%s_cor_avg" % (
                subject[:4],
                band,
            ))
        stc_pln = mne.read_source_estimate(
            beamformer_results + "%s_planning_%s_cor_avg" % (
                subject[:4],
                band,
            ))

        X_tmp = np.empty((2, stc_cls.data.shape[0], stc_cls.data.shape[1]))
        X_tmp[0, :] = stc_cls.data
        X_tmp[1, :] = stc_pln.data

        if j == 0:
            X = X_tmp
            y = np.array((0, 1))
        else:
            X = np.vstack((X, X_tmp))
            y = np.concatenate((y, np.array((0, 1))))

    X_y = dict(X=X, y=y)
    np.save(beamformer_mvpa + "X_cls_v_pln_%s.npy" % (band), X)

np.save(beamformer_mvpa + "y_cls_v_pln.npy", y)
