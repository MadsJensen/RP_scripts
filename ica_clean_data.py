#  -*- coding: utf-8 -*-
"""
Created on Wed Oct  8 14:45:02 2014.

@author: mje
"""
import mne
import sys

from mne.preprocessing import ICA, create_eog_epochs

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
        eog=False,
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

    eog_epochs = create_eog_epochs(epochs, ch_name="EOG002")  # TODO: check EOG
    eog_average = eog_epochs.average()
    # channel name
    eog_inds, scores = ica.find_bads_eog(epochs)

    eog_inds = eog_inds[:n_max_eog]
    ica.exclude += eog_inds

    fig = ica.plot_scores(
        scores, exclude=eog_inds, title=title % ('eog', subject))
    fig.savefig(ica_folder + "pics/%s_%s_eog_scores.png" % (subject, condition
                                                            ))
    fig = ica.plot_sources(eog_average, exclude=eog_inds)
    fig.savefig(ica_folder + "pics/%s_%s_eog_source.png" % (subject, condition
                                                            ))

    fig = ica.plot_components(
        eog_inds, title=title % ('eog', subject), colorbar=True)
    fig.savefig(ica_folder + "pics/%s_%s_eog_component.png" % (subject,
                                                               condition))
    fig = ica.plot_overlay(eog_average, exclude=eog_inds, show=False)
    fig.savefig(ica_folder + "pics/%s_%s_eog_excluded.png" % (subject,
                                                              condition))
    fig = ica.plot_properties(epochs, picks=eog_inds)
    fig.savefig(ica_folder + "pics/%s_%s_plot_properties.png" % (subject,
                                                                 condition))

    del eog_epochs, eog_average

    ##########################################################################
    # Apply the solution to Raw, Epochs or Evoked like this:
    epochs_ica = ica.apply(epochs)
    ica.save(ica_folder + "%s_%s-ica.fif" % (subject, condition))  # save ICA
    # componenets
    # Save raw with ICA removed
    epochs_ica.save(
        ica_folder + "%s_%s_ica-epo.fif" %
        (subject, condition),
        overwrite=True)
