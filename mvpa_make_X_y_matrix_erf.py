import numpy as np
import mne
import h5io
from my_settings import (erf_raw, erf_mvpa, erf_results, subjects, conditions,
                         make_rolling_mean, make_rolling_mean_stc)

tmp = mne.read_epochs(erf_raw + "0016_classic_ar_grads_erf-epo.fif")
data_shape = tmp.get_data().shape

windows_size = 10

for condition in conditions:
    X = np.zeros((len(subjects), data_shape[1],
                  data_shape[2] - (windows_size * 2)))

    for jj, subject in enumerate(subjects):
        epo = mne.read_epochs(erf_raw + "%s_%s_ar_grads_erf-epo.fif" %
                              (subject[:4], condition))

        X[jj] = make_rolling_mean(
            epo.average(),
            windows_size=windows_size)[:, windows_size:-windows_size]

    np.save(erf_mvpa + "X_%s_erf_RM.npy" % condition, X)

for condition in conditions:
    for j, subject in enumerate(subjects):
        stc_cls = mne.read_source_estimate(erf_results + "%s_classic_cor_avg" %
                                           (subject[:4], ))
        stc_pln = mne.read_source_estimate(
            erf_results + "%s_plannings_cor_avg" % (subject[:4]))

        X_tmp = np.empty((2, stc_cls.data.shape[0], stc_cls.data.shape[1]))
        X_tmp[0, :] = make_rolling_mean_stc(stc_cls, windows_size=windows_size)
        X_tmp[1, :] = make_rolling_mean_stc(stc_pln, windows_size=windows_size)

        if j == 0:
            X = X_tmp
            y = np.array((0, 1))
        else:
            X = np.vstack((X, X_tmp))
            y = np.concatenate((y, np.array((0, 1))))

    X_y = dict(X=X, y=y)
    h5io.write_hdf5(erf_mvpa + "X_cls_v_pln_.hd5", X_y)
