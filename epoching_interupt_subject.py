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

raw = mne.io.Raw(save_folder + "%s_interupt_filtered_ica_mc_tsss-raw.fif" % (
    subject))
events = mne.find_events(raw)

for j in range(len(events)):
    if events[:, 2][j] == 1 and events[:, 2][j - 1] == 2:
        events[:, 2][j] = 3

event_id = {'press': 1, "cue": 2, "cued_press": 3}

# Setup for reading the raw data
picks = mne.pick_types(raw.info,
                       meg=True,
                       eeg=False,
                       stim=True,
                       eog=True,
                       exclude='bads')
# Read epochs
epochs = mne.Epochs(raw,
                    events,
                    event_id,
                    tmin,
                    tmax,
                    picks=picks,
                    baseline=(None, -3.5),
                    reject=reject_params,
                    preload=True)

epochs.drop_bad(reject=reject_params)
fig = epochs.plot_drop_log(subject=subject, show=False)
fig.savefig(epochs_folder + "pics/%s_interupt_drop_log.png" % (subject))

# Save epochs
epochs.save(epochs_folder + "%s_interupt-epo.fif" % (subject))
