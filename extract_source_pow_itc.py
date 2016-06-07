# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 21:06:28 2016

@author: mje
"""
import mne
from mne.minimum_norm import (read_inverse_operator, source_induced_power)
import numpy as np
import sys
from my_settings import *

subject = sys.argv[1]

# Settings
method = "dSPM"
freqs = np.arange(6, 90, 3)  # define frequencies of interest
n_cycles = freqs / 3.  # different number of cycle per frequency
snr = 1.
lambda2 = 1. / snr**2

labels = mne.read_labels_from_annot(subject=subject,
                                    parc="PALS_B12_Brodmann",
                                    regexp="Brodmann")

for condition in conditions:
    inv = read_inverse_operator(mne_folder + "%s_%s-inv.fif" % (subject,
                                                                condition))
    epochs = mne.read_epochs(epochs_folder + "%s_%s-epo.fif" % (subject,
                                                                condition))
    # epochs.resample(500, n_jobs=1)
    power_lbl = np.empty([len(labels, ), len(freqs), len(epochs.times)])
    itc_lbl = np.empty([len(labels, ), len(freqs), len(epochs.times)])

    for j, label in enumerate(labels):
        print("\n****************************")
        print("Working on: %s" % label.name)
        print("****************************\n")
        
        power, itc = source_induced_power(epochs,
                                          inv,
                                          freqs,
                                          label,
                                          lambda2=lambda2,
                                          method=method,
                                          n_cycles=n_cycles,
                                          decim=2,
                                          pick_ori="normal",
                                          baseline=(-3.5, -3.2),
                                          baseline_mode='zscore',
                                          n_jobs=1,
                                          pca=True)
        power_lbl[j] = power.mean(axis=0)
        itc_lbl[j] = itc.mean(axis=0)

    np.save(source_folder + "source_TF/%s_%s_source-pow.npy" %
            (subject, condition), power)
    np.save(source_folder + "source_TF/%s_%s_source-itc.npy" %
            (subject, condition), itc)
