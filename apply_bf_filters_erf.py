import mne
import sys
import numpy as np
from scipy.signal import hilbert
from mne.beamformer import apply_lcmv_epochs
from my_settings import (erf_filters, erf_raw, erf_results, conditions)


def compute_source_itc(stcs):
    n_trials = len(stcs)

    tmp = np.zeros(stcs[0].data.shape, dtype=np.complex)
    for stc in stcs:
        # divide by amplitude and sum angles
        tmp += stc.data / abs(stc.data)

    # take absolute value and normalize
    itc = abs(tmp) / n_trials

    return itc


subject = sys.argv[1]

filters = np.load(erf_filters + "%s_bf_filters_cor.npy" %
                  (subject[:4])).item()

for condition in conditions:
    epochs = mne.read_epochs(erf_raw + "%s_%s_ar_grads-epo.fif" %
                             (subject[:4], condition))
    epochs.pick_types(meg="grad")
    epochs_hilb = epochs.copy()
    epochs_hilb._data = hilbert(epochs.get_data())

    itcs = []  # ITC across freq bands
    stcs = apply_lcmv_epochs(
        epochs_hilb, filters=filters, max_ori_out="signed")

    itcs.append(compute_source_itc(stcs))
    stc_dummy = stcs[0]
    stc_dummy.data = itcs[0]
    stc_dummy.save(erf_results + "%s_%s_cor" % (subject[:4], condition))

del epochs, stcs, stc_dummy
