import sys

import mne
import numpy as np
from mne.beamformer import make_lcmv
from mne.utils import estimate_rank
from scipy.signal import hilbert

from my_settings import (beamformer_filters, beamformer_raw, beamformer_source,
                         bands)

subject = sys.argv[1]

fwd = mne.read_forward_solution(
    beamformer_source + "%s_avg_cor-fwd.fif" % subject[:4])

for band in bands:
    epochs_cls = mne.read_epochs(
        beamformer_raw + "%s_classic_%s_ar_grads-epo.fif" % (subject[:4],
                                                             band))
    epochs_pln = mne.read_epochs(
        beamformer_raw + "%s_planning_%s_ar_grads-epo.fif" % (subject[:4],
                                                              band))
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

    np.save(beamformer_filters + "%s_optim_%s_bf_filters_cor.npy" %
            (subject[:4], band), filters)
