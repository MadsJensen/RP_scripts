import mne
import numpy as np
import sys
from my_settings import (conditions, save_folder, reject_params, maxfiltered_folder_)

from mne.utils import check_random_state
from autoreject import (LocalAutoRejectCV, compute_thresholds,
                        set_matplotlib_defaults)

import matplotlib
# matplotlib.use('Agg')


subject = sys.argv[1]

# SETTINGS
tmin, tmax = -4, 1
n_interpolates = np.array([1, 4, 32])
consensus_percs = np.linspace(0, 1.0, 11)

for condition in conditions[:2]:
    raw = mne.io.Raw(save_folder + "%s_%s_filtered_ica_mc_tsss-raw.fif" % (
        subject, condition))

    # Setup events
    events = mne.find_events(raw)

    if condition is "interupt":
        event_id = {'press': 1, "about_to_press": 2}
    else:
        event_id = {'press': 1}

    # And pick MEG channels for repairing. Currently, :mod:`autoreject` can repair
    # only one channel type at a time.

    ###############################################################################
    raw.info['bads'] = []

    ###############################################################################
    # Now, we can create epochs. The ``reject`` params will be set to ``None``
    # because we do not want epochs to be dropped when instantiating
    # :class:`mne.Epochs`.

    ###############################################################################
    raw.info['projs'] = list()  # remove proj, don't proj while interpolating

    # Setup for reading the raw data
    picks = mne.pick_types(
        raw.info, meg=True, eeg=False, stim=False, eog=False, exclude='bads')
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


    ###############################################################################
    from functools import partial
    thresh_func = partial(compute_thresholds, method='random_search',
                          random_state=42)

    ###############################################################################
    # :class:`autoreject.LocalAutoRejectCV` internally does cross-validation to
    # determine the optimal values :math:`\rho^{*}` and :math:`\kappa^{*}`

    ###############################################################################

    ar = LocalAutoRejectCV(n_interpolates, consensus_percs,
                           thresh_func=thresh_func)
    epochs_clean = ar.fit_transform(epochs)

    evoked = epochs.average()
    evoked_clean = epochs_clean.average()
