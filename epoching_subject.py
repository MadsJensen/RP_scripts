"""Epoch a raw data set.

"""
import mne
import sys

from my_settings import *

subject = sys.argv[1]

# SETTINGS
tmin, tmax = -4, 1
event_id = {'press': 1}

for condition in conditions:
    raw = mne.io.Raw(save_folder + "%s_%s_filtered_ica_mc_tsss-raw.fif"
                     % (subject, condition))
    events = mne.find_events(raw)
    # Setup for reading the raw data
    picks = mne.pick_types(raw.info, meg=False, eeg=True,
                           stim=True, eog=True, exclude='bads')
    # Read epochs
    epochs = mne.Epochs(raw, events, event_id, tmin, tmax, picks=picks,
                        baseline=(None, -3.5), reject=reject_params,
                        preload=True)
    # Save epochs
    epochs.save(epochs_folder + "%s_%s-epo.fif" % (subject, condition))
