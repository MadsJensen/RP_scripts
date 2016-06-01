#  -*- coding: utf-8 -*-
"""
Created on Wed Oct  8 14:45:02 2014.

@author: mje
"""

%reset -f

import mne
import sys

from mne.io import Raw
from mne.preprocessing import ICA,  create_eog_epochs

import matplotlib
matplotlib.use('Agg')

#from my_settings import *

data_path = '/home/mje/mnt/hyades/scratch2/MINDLAB2011_24-MEG-readiness/mne_analysis_new/'

save_folder = data_path + "filter_ica_data/"
maxfiltered_folder = data_path + "maxfiltered_data/"

# SETTINGS
n_jobs = 1
l_freq, h_freq = 1, 98  # High and low frequency setting for the band pass
n_freq = 50  # notch filter frequency
decim = 4  # decim value

matplotlib.pyplot.close("all")

subject = "0020"
condition = "interupt"

reject_params = dict(grad=4000e-13,  # T / m (gradiometers)
                     mag=4e-12  # T (magnetometers)
                     )

# SETTINGS

#raw = Raw(save_folder + "%s_%s_filtered_mc_tsss-raw.fif" % (subject,
#                                                            condition),
#          preload=True)

raw = Raw(maxfiltered_folder + "%s_%s_mc_tsss-raw.fif" % (subject,
                                                          condition),
          preload=True)
raw.drop_channels(raw.info["bads"])

raw.notch_filter(n_freq, n_jobs=n_jobs)
raw.filter(l_freq, h_freq, n_jobs=n_jobs)

raw.save(save_folder + "%s_%s_filtered_mc_tsss-raw.fif" % (subject,
                                                           condition),
         overwrite=True)



# ICA Part
ica = ICA(n_components=0.99, method='fastica', max_iter=256)

picks = mne.pick_types(raw.info, meg=True, eeg=False, eog=False, emg=False,
                       bio=False, stim=False, exclude='bads')

ica.fit(raw, picks=picks, decim=decim, reject=reject_params)

# maximum number of components to reject
n_max_eog = 1

##########################################################################
# 2) identify bad components by analyzing latent sources.

# DETECT EOG BY CORRELATION
# HORIZONTAL EOG
title = "ICA: %s for %s"
picks = mne.pick_types(raw.info, meg=True, eeg=False, eog=False, emg=False,
                       bio=True, stim=False, exclude='bads')
eog_epochs = create_eog_epochs(raw, ch_name="EOG001")  
eog_average = eog_epochs.average()
# channel name
eog_inds, scores = ica.find_bads_eog(raw)

# ica.plot_components()
ica.plot_sources(raw)

 # %%
# eog_inds = [10]
# ica.exclude += eog_inds

fig = ica.plot_scores(scores, exclude=eog_inds,
                      title=title % ('eog', subject))
fig.savefig(save_folder + "pics/%s_%s_eog_scores.png" % (subject,
                                                         condition))
                                                         

fig = ica.plot_sources(eog_average, exclude=None)
fig.savefig(save_folder + "pics/%s_%s_eog_source.png" % (subject,
                                                         condition))

fig = ica.plot_components(ica.exclude, title=title % ('eog', subject),
                          colorbar=True)
fig.savefig(save_folder + "pics/%s_%s_eog_component.png" % (subject,
                                                            condition))
fig = ica.plot_overlay(eog_average, exclude=None, show=False)                                                                
fig.savefig(save_folder + "pics/%s_%s_eog_excluded.png" % (subject,
                                                            condition))


# del eog_epochs, eog_average

##########################################################################
# Apply the solution to Raw, Epochs or Evoked like this:
raw_ica = ica.apply(raw)
ica.save(save_folder + "%s_%s-ica.fif" % (subject, condition))  # save ICA
# componenets
# Save raw with ICA removed
raw_ica.save(save_folder + "%s_%s_filtered_ica_mc_tsss-raw.fif" % (
    subject, condition),
             overwrite=True)
