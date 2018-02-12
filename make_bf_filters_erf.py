import sys

import mne
import numpy as np
from mne.beamformer import make_lcmv
from mne.utils import estimate_rank
from scipy.signal import hilbert

from my_settings import (erf_filters, erf_raw, beamformer_source)

subject = sys.argv[1]

fwd = mne.read_forward_solution(
    beamformer_source + "%s_avg_cor-fwd.fif" % subject[:4])

epochs_cls = mne.read_epochs(
    erf_raw + "%s_classic_ar_grads_erf-epo.fif" % (subject[:4]))
epochs_pln = mne.read_epochs(
    erf_raw + "%s_planning_ar_grads_erf-epo.fif" % (subject[:4]))
epochs_pln.info["dev_head_t"] = epochs_cls.info["dev_head_t"]

epochs = mne.concatenate_epochs([epochs_cls, epochs_pln])

epochs.pick_types(meg="grad")
epochs_hilb = epochs.copy()
epochs_hilb._data = hilbert(epochs.get_data())
data_cov = mne.compute_covariance(epochs, tmin=None, tmax=None)

rank_cov = estimate_rank(data_cov['data'], tol='auto')

filters = make_lcmv(
    epochs.info,
    fwd,
    data_cov=data_cov,
    pick_ori='max-power',
    weight_norm='nai',
    reg=0.0)

np.save(erf_filters + "%s_bf_filters_cor.npy" % (subject[:4]),
        filters)
