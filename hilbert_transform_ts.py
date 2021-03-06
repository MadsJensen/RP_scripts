# -*- coding: utf-8 -*-
"""
Created on Mon Jun 13 12:50:46 2016

@author: mje
"""
import mne
from scipy.signal import hilbert
import numpy as np
import sys

from my_settings import (conditions, source_folder, bands)

sfreq = 1000

subject = sys.argv[1]

result = {}

for condition in conditions:
    ts = np.load(source_folder + "ave_ts/%s_%s_ts-epo.npy" % (subject,
                                                              condition))
    for band in bands.keys():
        data = mne.filter.filter_data(ts, sfreq, bands[band][0],
                                      bands[band][1])
        ht_data = hilbert(data)
        result[band] = ht_data

        np.save(source_folder + "hilbert_data/%s_%s_ht-epo.npy" %
                (subject, condition), result)
