import h5io
import mne
import numpy as np

from my_settings import (beamformer_results, beamformer_mvpa, bands,
                         subjects, make_rolling_mean_stc)

for band in bands:
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
        X_tmp[0, :] = make_rolling_mean_stc(stc_cls)
        X_tmp[1, :] = make_rolling_mean_stc(stc_pln)

        if j == 0:
            X = X_tmp
            y = np.array((0, 1))
        else:
            X = np.vstack((X, X_tmp))
            y = np.concatenate((y, np.array((0, 1))))

    X_y = dict(X=X, y=y)
    h5io.write_hdf5(beamformer_mvpa + "Xy_cls_v_pln_%s_RM.hd5" % band, X_y)
