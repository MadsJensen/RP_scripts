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
    tfr = cwt_morlet(ts,
                     sfreq=sfreq,
                     frequencies=frequencies,
                     use_fft=True,
                     n_cycles=n_cycles)
