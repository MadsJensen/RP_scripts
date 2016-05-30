"""Epoch a raw data set.

"""
import mne
import sys
import matplotlib
matplotlib.use('Agg')

from my_settings import *

subject = sys.argv[1]

# SETTINGS
tmin, tmax = -4, 1

for condition in conditions:
    raw = mne.io.Raw(save_folder + "%s_%s_filtered_ica_mc_tsss-raw.fif"
                     % (subject, condition))
    events = mne.find_events(raw)

    if condition is "interupt":
        event_id = {'press': 1,
                    "about_to_press": 2}
    else:
        event_id = {'press': 1}


    # Setup for reading the raw data
    picks = mne.pick_types(raw.info, meg=True, eeg=False,
                           stim=True, eog=True, exclude='bads')
    # Read epochs
    epochs = mne.Epochs(raw, events, event_id, tmin, tmax, picks=picks,
                        baseline=(None, -3.5), reject=reject_params,
                        preload=True)

    epochs.drop_bad(reject=reject_params)
    fig = epochs.plot_drop_log(subject=subject, show=False)
    fig.savefig(epochs_folder + "pics/%s_%s_drop_log.png" % (subject,
                                                             condition))

    # Save epochs
    epochs.save(epochs_folder + "%s_%s-epo.fif" % (subject, condition))
