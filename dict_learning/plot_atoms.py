from utils_function import plot_atoms
import sys
from my_settings import (erf_raw, dict_learning, conditions)

import matplotlib.pyplot as plt
import numpy as np
import mne
import joblib

subject = sys.argv[1]
epo = mne.read_epochs(erf_raw + "0008_classic_ar_grads_erf_hg-epo.fif")
info = epo.info
t = epo.times[::2]

conditions = list(conditions.keys())  # only need the keys

atom_sets = [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9], [10, 11, 12, 13, 14],
             [15, 16, 17, 18, 19]]

for condition in conditions:
    cdl = joblib.load(
        dict_learning +
        '{}_{}_ar_grads_std_csc.jbl'.format(subject[:4], condition))

    for jj, atom_set in enumerate(atom_sets):
        fig = plot_atoms(cdl, plotted_atoms=atom_set)
        plt.title("subject: {}; condition: {}".format(subject, condition))
        fig.tight_layout()
        plt.savefig(
            dict_learning +
            'plots/{}_{}_std_atoms_{}.png'.format(subject[:4], condition, jj))
        plt.close('all')
