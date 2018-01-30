#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 14:40:18 2017

@author: mje
"""

import sys
import numpy as np
import mne  # noqa
from autoreject import (LocalAutoRejectCV, compute_thresholds)
from functools import partial  # noqa
from my_settings import beamformer_raw, conditions
from stormdb.access import Query
import pickle

subject = sys.argv[1]

proj_name = 'MINDLAB2011_24-MEG-readiness'
qr = Query(proj_name, verbose=False)
study = qr.get_studies(subj_id=subject, modality="MEG")[0]

random_state = 3423534231

n_interpolates = np.array([1, 4, 32])
consensus_percs = np.linspace(0, 1.0, 11)
n_jobs = 1
l_freq, h_freq = 1, 95  # High and low frequency setting for the band pass
n_freq = 50  # notch filter frequency
decim = 4  # decim value

event_id = {"press": 1}

tmin, tmax = -3.5, 0.5
baseline = None  # (-0.1, 0)

iter_freqs = [('Alpha', 8, 12), ('Beta', 13, 30), ('Gamma_low', 30, 45),
              ('Gamma_high_1', 55, 70), ('Gamma_high_2', 70, 90)]

for condition in conditions.keys():
    for band, fmin, fmax in iter_freqs:
        series = qr.get_files(
            subj_id=subject,
            study=study,
            modality="MEG",
            series=conditions[condition])

        raw = mne.io.read_raw_fif(series[0], preload=True)
        raw.info['bads'] = []
        picks = mne.pick_types(
            raw.info,
            meg=True,
            eeg=False,
            stim=False,
            eog=False,
            include=[],
            exclude=[])

        raw.info['projs'] = list()  # remove proj

        raw.resample(500, n_jobs=1)
        raw.notch_filter(n_freq, fir_design='firwin', n_jobs=n_jobs)
        raw.filter(fmin, None, fir_design='firwin', n_jobs=n_jobs)
        raw.filter(None, fmax, fir_design='firwin', n_jobs=n_jobs)

        events = mne.find_events(
            raw,
            stim_channel='STI101',
            shortest_event=2,
            min_duration=0.004,
            verbose=True)
        # adjust time delay from tubes
       
        # TODO:
        epochs = mne.Epochs(
            raw,
            events,
            event_id,
            tmin,
            tmax,
            baseline=(None, -3),
            picks=picks,
            reject=None,
            verbose=False,
            detrend=0,
            preload=True)

        thresh_func = partial(
            compute_thresholds,
            picks=picks,
            method='random_search',
            random_state=random_state)

        ar = LocalAutoRejectCV(
            n_interpolates, consensus_percs, thresh_func=thresh_func)
        ar.fit(epochs)

        epochs_clean = ar.transform(epochs)
        epochs_clean.save(beamformer_raw + "%s_%s_%s_ar-epo.fif" %
                          (subject[:4], condition, band))
        pickle.dump(ar,
                    open(beamformer_raw + "%s_%s_%s_ar.pkl" %
                         (subject[:4], condition, band), "wb"))
