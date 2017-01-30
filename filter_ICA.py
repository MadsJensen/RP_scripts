#  -*- coding: utf-8 -*-
"""
Created on Wed Oct  8 14:45:02 2014.

@author: mje
"""
import mne
import sys

from mne.io import Raw
from mne.preprocessing import ICA, create_eog_epochs, create_ecg_epochs

import matplotlib
matplotlib.use('Agg')

from my_settings import *

subject = sys.argv[1]

# SETTINGS
n_jobs = 1
l_freq, h_freq = 1, 95  # High and low frequency setting for the band pass
n_freq = 50  # notch filter frequency
decim = 4  # decim value

for condition in conditions:
    raw = Raw(mf_autobad_off_folder + "%s_%s_mc_tsss-raw.fif" %
              (subject, condition),
              preload=True)
    raw.drop_channels(raw.info["bads"])

    raw.notch_filter(n_freq, n_jobs=n_jobs)
    raw.filter(l_freq, None, n_jobs=n_jobs)

    raw.save(
        mf_autobad_off_folder + "%s_%s_filtered_mc_tsss-raw.fif" %
        (subject, condition),
        overwrite=True)

    # ICA Part
    ica = ICA(n_components=0.99, method='fastica', max_iter=512)

    picks = mne.pick_types(
        raw.info,
        meg=True,
        eeg=False,
        eog=False,
        emg=False,
        stim=False,
        exclude='bads')

    ica.fit(raw, picks=picks, decim=decim, reject=reject_params)

    # maximum number of components to reject
    n_max_eog = 1
    n_max_ecg = 3

    ##########################################################################
    # 2) identify bad components by analyzing latent sources.

    # DETECT EOG BY CORRELATION
    # HORIZONTAL EOG
    title = "ICA: %s for %s"

    eog_epochs = create_eog_epochs(raw, ch_name="EOG002")  # TODO: check EOG
    eog_average = eog_epochs.average()
    # channel name
    eog_inds, scores = ica.find_bads_eog(raw)

    eog_inds = eog_inds[:n_max_eog]
    ica.exclude += eog_inds

    if eog_inds:
        fig = ica.plot_scores(
            scores, exclude=eog_inds, title=title % ('eog', subject))
        fig.savefig(ica_folder + "plots/%s_%s_eog_scores_2.png" % (subject,
                                                                   condition))
        fig = ica.plot_sources(eog_average, exclude=eog_inds)
        fig.savefig(ica_folder + "plots/%s_%s_eog_source_2.png" % (subject,
                                                                   condition))

        fig = ica.plot_components(
            eog_inds, title=title % ('eog', subject), colorbar=True)
        fig.savefig(ica_folder + "plots/%s_%s_eog_component_2.png" % (
            subject, condition))
        fig = ica.plot_overlay(eog_average, exclude=eog_inds, show=False)
        fig.savefig(ica_folder + "plots/%s_%s_eog_excluded_2.png" % (
            subject, condition))

    del eog_epochs, eog_average

    # ECG
    ecg_epochs = create_ecg_epochs(raw, tmin=-.5, tmax=.5)
    ecg_inds, scores = ica.find_bads_ecg(ecg_epochs)
    ecg_inds = ecg_inds[:n_max_ecg]
    ica.exclude.extend(ecg_inds)

    if ecg_inds:
        fig = ica.plot_components(
            ecg_inds, title=title % ('ecg', subject), colorbar=True)
        fig.savefig(ica_folder + "plots/%s_%s_ecg_component_2.png" % (
            subject, condition))
        fig = ica.plot_overlay(raw, exclude=ecg_inds, show=False)
        fig.savefig(ica_folder + "plots/%s_%s_ecg_excluded_2.png" % (
            subject, condition))
        fig = ica.plot_properties(raw, picks=ecg_inds)
        fig[0].savefig(ica_folder + "plots/%s_%s_plot_properties_2.png" % (
            subject, condition))

    ##########################################################################
    # Apply the solution to Raw, Epochs or Evoked like this:
    raw_ica = ica.apply(raw)
    ica.save(ica_folder + "%s_%s-ica_2.fif" % (subject, condition))  # save ICA
    # componenets
    # Save raw with ICA removed
    raw_ica.save(
        ica_folder + "%s_%s_ica-raw.fif" % (subject, condition),
        overwrite=True)
