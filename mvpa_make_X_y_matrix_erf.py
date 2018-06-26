import numpy as np
import mne
from my_settings import (erf_raw, erf_mvpa, subjects, conditions)

tmp = mne.read_epochs(erf_raw + "0016_classic_ar_grads_erf-epo.fif")
data_shape = tmp.get_data().shape

for condition in conditions:
    X = np.zeros((len(subjects), data_shape[1], data_shape[2]))

    for jj, subject in enumerate(subjects):
        epo = mne.read_epochs(erf_raw + "%s_%s_ar_grads_erf-epo.fif" %
                              (subject[:4], condition))

        X[jj] = epo.average().data

    np.save(erf_mvpa + "X_%s_erf.npy" % condition, X)
