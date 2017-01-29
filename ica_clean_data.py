#  -*- coding: utf-8 -*-
"""
Created on Wed Oct  8 14:45:02 2014.

@author: mje
"""
import mne
import sys

from mne.preprocessing import ICA

from my_settings import (epochs_folder, conditions, ica_folder)

import matplotlib
matplotlib.use('Agg')

subject = sys.argv[1]

# SETTINGS
n_jobs = 1
decim = 4  # decim value

for condition in conditions:
    epochs = mne.read_epochs(epochs_folder + "%s_%s_ar-epo.fif" % (subject,
                                                                   condition))

    # ICA Part
    ica = ICA(n_components=0.99, method='fastica', max_iter=256)

    picks = mne.pick_types(
        epochs.info,
        meg=True,
        eeg=False,
        eog=True,
        emg=False,
        stim=False,
        exclude=[])

    ica.fit(epochs, picks=picks, decim=decim, reject=None)

    # maximum number of components to reject
    n_max_eog = 1

    ##########################################################################
    # 2) identify bad components by analyzing latent sources.

    # DETECT EOG BY CORRELATION
    # HORIZONTAL EOG
    title = "ICA: %s for %s"

    # channel name
    eog_inds, scores = ica.find_bads_eog(epochs)

    eog_inds = eog_inds[:n_max_eog]
    ica.exclude += eog_inds

    if eog_inds:
        fig = ica.plot_scores(
            scores, exclude=eog_inds, title=title % ('eog', subject))
        fig.savefig(ica_folder + "plots/%s_%s_eog_scores.png" % (subject, condition
                                                                ))
        fig = ica.plot_sources(epochs, exclude=eog_inds)
        fig.savefig(ica_folder + "plots/%s_%s_eog_source.png" % (subject, condition
                                                                ))
        
        fig = ica.plot_components(
            eog_inds, title=title % ('eog', subject), colorbar=True)
        fig.savefig(ica_folder + "plots/%s_%s_eog_component.png" % (subject,
                                                                   condition))
        fig = ica.plot_overlay(epochs.average(), exclude=eog_inds, show=False)
        fig.savefig(ica_folder + "plots/%s_%s_eog_excluded.png" % (subject,
                                                                  condition))
        fig = ica.plot_properties(epochs, picks=eog_inds)
        fig[0].savefig(ica_folder + "plots/%s_%s_plot_properties.png" % (subject,
                                                                     condition))

    ## ECG
    ecg_inds, scores = ica.find_bads_ecg(epochs)
    ica.exclude += ecg_inds

    if ecg_inds:    
        fig = ica.plot_components(
            ecg_inds, title=title % ('ecg', subject), colorbar=True)
        fig.savefig(ica_folder + "plots/%s_%s_ecg_component.png" % (subject,
                                                                   condition))
        fig = ica.plot_overlay(epochs.average(), exclude=ecg_inds, show=False)
        fig.savefig(ica_folder + "plots/%s_%s_ecg_excluded.png" % (subject,
                                                                  condition))
        fig = ica.plot_properties(epochs, picks=ecg_inds)
        fig[0].savefig(ica_folder + "plots/%s_%s_plot_properties.png" % (subject,
                                                                     condition))


    ##########################################################################
    # Apply the solution to Raw, Epochs or Evoked like this:
    epochs_ica = ica.apply(epochs)
    ica.save(ica_folder + "%s_%s-ica.fif" % (subject, condition))  # save ICA
    # componenets
    # Save raw with ICA removed
    epochs_ica.save(
        ica_folder + "%s_%s_ar_ica-epo.fif" %
        (subject, condition))
