import sys
import cPickle as pickle

import mne
import numpy as np

from my_settings import (mne_folder, conditions, source_folder, epochs_folder)

subject = sys.argv[1]

labels = mne.read_labels_from_annot(
    subject=subject, parc="PALS_B12_Brodmann", regexp="Brodmann")

time_conditions = ["pre", "preact"]

for time_cond in time_conditions:
    for condition in conditions:
        
    stcs = pickle.load(open(source_folder +
                "%s_%s_psd_pre_epo.pkl" % (subject, condition), 'rb'))
    


