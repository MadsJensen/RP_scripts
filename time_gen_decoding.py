import sys
import numpy as np
import mne
from mne.decoding import GeneralizationAcrossTime
from sklearn.externals import joblib

from my_settings import (epochs_folder, data_path)

import matplotlib
matplotlib.use('Agg')

subject = sys.argv[1]

# Load epochs from both conditions
epochs_classic = mne.read_epochs(epochs_folder + "%s_classic-epo.fif" % (
    subject))
epochs_plan = mne.read_epochs(epochs_folder + "%s_plan-epo.fif" % (subject))

# Fix the events for the plan epochs so they can be concatenated
epochs_plan.event_id["press"] = 2
epochs_plan.event_id["plan"] = epochs_plan.event_id.pop("press")
epochs_plan.events[:, 2] = 2

# Equalise channels and epochs, and concatenate epochs
mne.equalize_channels([epochs_classic, epochs_plan])
mne.epochs.equalize_epoch_counts([epochs_classic, epochs_plan])

# Dirty hack # TODO: Check this from the Maxfilter side
epochs_classic.info['dev_head_t'] = epochs_plan.info['dev_head_t']

epochs = mne.concatenate_epochs([epochs_classic, epochs_plan])
# Crop and downsmample to make it faster
epochs.crop(tmin=-3.5, tmax=0)
epochs.resample(250)

# Setup the y vector and GAT
y = np.concatenate(
    (np.zeros(len(epochs["press"])), np.ones(len(epochs["plan"]))))
gat = GeneralizationAcrossTime(
    predict_mode='mean-prediction', scorer="ruc_aoc", n_jobs=1)

# Fit model
gat.fit(epochs, y=y)

# Scoring and visualise result
gat.score(epochs, y=y)

# Save model
joblib.dump(gat, data_path + "decode_time_gen/%s_gat.jl" % subject)

fig = gat.plot(
    title="Temporal Gen (Classic vs planning): left to right sub: %s" %
    subject)
fig.savefig(data_path + "decode_time_gen/%s_gat_matrix.png" % subject)
