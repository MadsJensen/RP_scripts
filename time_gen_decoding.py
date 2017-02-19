import sys
import numpy as np
import mne
from mne.decoding import GeneralizationAcrossTime

from my_settings import (epochs_folder, conditions)

subject = sys.argv[1]

epochs_classic = mne.read_epochs(epochs_folder + "%s_classic-epo.fif" % (
    subject))
epochs_plan = mne.read_epochs(epochs_folder + "%s_plan-epo.fif" % (subject))

epochs = mne.concatenate_epochs([epochs_classic, epochs_plan])

gat = GeneralizationAcrossTime(predict_mode='mean-prediction', n_jobs=1)
gat.fit(epochs[('AudL', 'VisL')], y=viz_vs_auditory_l)

gat.score(epochs, y=viz_vs_auditory_r)
gat.plot(title="Temporal Generalization (visual vs auditory): left to right")
