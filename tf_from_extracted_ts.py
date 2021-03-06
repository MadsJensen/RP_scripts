import numpy as np
import sys

from mne.time_frequency import cwt_morlet

from my_settings import *

subject = sys.argv[1]

sfreq = 500
frequencies = np.arange(4, 90, 2)
n_cycles = frequencies / 3.

for condition in conditions:
    ts = np.load(source_folder + "ave_ts/%s_%s_ts-epo.npy" % (subject,
                                                              condition))

    tfr_all = np.empty(
        [ts.shape[0], ts.shape[1], frequencies.shape[0], ts.shape[2]],
        dtype="complex128")
    for j, t in enumerate(ts):
        tfr = cwt_morlet(t,
                         sfreq=sfreq,
                         freqs=frequencies,
                         use_fft=True,
                         n_cycles=n_cycles)
        tfr_all[j] = tfr
    np.save(source_folder + "source_TF/%s_%s_tfr-extract.npy" % (subject,
                                                                 condition),
                                                                 tfr_all)