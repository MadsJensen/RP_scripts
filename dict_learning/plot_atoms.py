from utils_function import plot_atoms
import sys
from my_settings import (erf_raw, dict_learning, conditions)

import matplotlib.pyplot as plt
import numpy as np
import mne
import joblib

subject = sys.argv[1]
epo = mne.read_epochs(erf_raw + "0008_classic_ar_grads_erf-epo.fif")
info = epo.info
t = epo.times[::2]

conditions = list(conditions.keys())  # only need the keys

for condition in conditions:
    cdl = joblib.load(dict_learning +
                      '{}_{}_ar_grads_csc.jbl'.format(subject, condition))

    fig = plot_atoms(cdl, plotted_atoms=[0, 1, 2, 3, 4])

    plt.savefig(dict_learning +
                'plots/{}_{}_atoms.png'.format(subject, condition))
