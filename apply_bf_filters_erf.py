import mne
import sys
import numpy as np
from mne.beamformer import apply_lcmv
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

filters = np.load(erf_filters + "%s_bf_filters_cor.npy" % (subject[:4])).item()

for condition in conditions:
    epochs = mne.read_epochs(erf_raw + "%s_%s_ar_grads_erf-epo.fif" %
                             (subject[:4], condition))
    epochs.pick_types(meg="grad")
    ave = epochs.average()

    stc = apply_lcmv(ave, filters=filters, max_ori_out="signed")

    stc.save(erf_results + "%s_%s_cor" % (subject[:4], condition))

del epochs, stc
