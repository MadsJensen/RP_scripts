# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 21:06:28 2016

@author: mje
"""
import mne
from mne.minimum_norm import (read_inverse_operator,
                              source_induced_power)
import numpy as np
import sys
from my_settings import *

subject = sys.argv[1]

# Settings
method = "dSPM"
snr = 3.
lambda2 = 1. / snr ** 2

labels = mne.read_labels_from_annot(subject=subject, parc="PALS_B12_Brodmann",
                                    regexp="Brodmann")


for condition in conditions:
    inv = read_inverse_operator(mne_folder + "%s_%s-inv.fif" % (subject,
                                                                condition))
    epochs = mne.read_epochs(epochs_folder + "%s_%s_-epo.fif" % (subject,
                                                                 condition))
    epochs.resample(500, n_jobs=4)

    freqs = np.arange(6, 90, 3)  # define frequencies of interest
    label = labels[52]
    n_cycles = frequencies / 3.  # different number of cycle per frequency

    power, itc = source_induced_power(epochs, inv, freqs,
                                      label, method=method,
                                      n_cycles=n_cycles,
                                      baseline=(-3.5, -3.2),
                                      baseline_mode='zscore',
                                      n_jobs=1, pca=True)

    np.save(source_folder + "%s_%s_source-pow.npy" % (subject, condition),
            power)
    np.save(source_folder + "%s_%s_source-itc.npy" % (subject, condition),
            itc)
