import mne
import numpy as np
import sys
from my_settings import (conditions, mf_autobad_off_folder, epochs_folder)
# from mne.utils import check_random_state
from autoreject import (LocalAutoRejectCV, compute_thresholds)

# import matplotlib
# matplotlib.use('Agg')

subject = sys.argv[1]

# SETTINGS
tmin, tmax = -4, 1
n_interpolates = np.array([1, 4, 32])
consensus_percs = np.linspace(0, 1.0, 11)

for condition in conditions:
    raw = mne.io.Raw(
        mf_autobad_off_folder + "%s_%s_mc_tsss-raw.fif" % (subject, condition),
        preload=True)
    raw.filter(1, None)
    raw.notch_filter(50)
    raw.filter(None, 95)

    # Setup events
    events = mne.find_events(raw)
    event_id = {'press': 1}

    # And pick MEG channels for repairing. Currently, :mod:`autoreject` can
    # repair only one channel type at a time.

    #########################################################################
    raw.info['bads'] = []

    ########################################################################
    # Now, we can create epochs. The ``reject`` params will be set to ``None``
    # because we do not want epochs to be dropped when instantiating
    # :class:`mne.Epochs`.

    #########################################################################
    raw.info['projs'] = list()  # remove proj, don't proj while interpolating

    # Setup for reading the raw data
    picks = mne.pick_types(
        raw.info, meg=True, eeg=False, stim=False, eog=True, exclude=[])
    # Read epochs
    epochs = mne.Epochs(
        raw,
        events,
        event_id,
        tmin,
        tmax,
        picks=picks,
        baseline=(None, -3.5),
        reject=None,
        detrend=0,
        preload=True)

    epochs = epochs["press"]
    #######################################################################
    from functools import partial
    thresh_func = partial(
        compute_thresholds, method='random_search', random_state=42)

    ######################################################################
    # :class:`autoreject.LocalAutoRejectCV` internally does cross-validation to
    # determine the optimal values :math:`\rho^{*}` and :math:`\kappa^{*}`

    #####################################################################
    epochs_grad = epochs.copy().pick_types(meg="grad")
    epochs_mag = epochs.copy().pick_types(meg="mag")

    ar_grad = LocalAutoRejectCV(
        n_interpolates,
        consensus_percs,
        thresh_func=thresh_func,
        verbose="progressbar")

    ar_mag = LocalAutoRejectCV(
        n_interpolates,
        consensus_percs,
        thresh_func=thresh_func,
        verbose="progressbar")

    epochs_grad_clean = ar_grad.fit_transform(epochs_grad)
    epochs_mag_clean = ar_mag.fit_transform(epochs_mag)

    epochs_clean = epochs.copy()

    bads_grads = ar_grad.bad_epochs_idx
    bads_mags = ar_mag.bad_epochs_idx
    bads_comb = list(set(list(bads_mags) + list(bads_grads)))
    bads_comb.sort()

    bad_idx_comb = np.zeros(len(epochs_clean.get_data()), dtype="bool")
    bad_idx_comb[bads_comb] = True

    mag_idx = mne.pick_types(epochs_clean.info, meg="mag")
    grad_idx = mne.pick_types(epochs_clean.info, meg="grad")

    epochs_clean.drop(bad_idx_comb)

    # Loop to drop bad epochs not present in the channel type
    # k is a counter to keep the index right after dropping epochs
    k = 0
    for i, j in enumerate(bads_comb):
        tmp = epochs_grad_clean.drop_log[j]
        if not tmp:
            epochs_grad_clean.drop(j - k)
            k += 1

    # Loop to drop bad epochs not present in the channel type
    # k is a counter to keep the index right after dropping epochs
    k = 0
    for i, j in enumerate(bads_comb):
        tmp = epochs_mag_clean.drop_log[j]
        if not tmp:
            epochs_mag_clean.drop(j - k)
            k += 1

    epochs_clean._data[:, grad_idx, :] = epochs_grad_clean.get_data()
    epochs_clean._data[:, mag_idx, :] = epochs_mag_clean.get_data()

    # Save epochs
    epochs_clean.save(epochs_folder + "%s_%s_ar-epo.fif" % (subject, condition
                                                            ))
