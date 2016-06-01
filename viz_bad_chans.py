# -*- coding: utf-8 -*-
"""
Created on Fri May 27 16:21:58 2016

@author: mje
"""

%reset -f

import mne
# from my_settings import *

maxfiltered_folder = '/home/mje/mnt/hyades/scratch2/MINDLAB2011_24-MEG-readiness/mne_analysis_new/maxfiltered_data/'

subject = "0020"
condition = "classic"

raw = mne.io.Raw(maxfiltered_folder + "%s_%s_mc_tsss-raw.fif" % (subject,
                                                                 condition),
                                                                 preload=True)
raw.plot(highpass=0.1, lowpass=40)

# %%
raw.save(maxfiltered_folder + "%s_%s_mc_tsss-raw.fif" % (subject, condition),
         overwrite=True)
         
# %%
condition = "plan"
raw = mne.io.Raw(maxfiltered_folder + "%s_%s_mc_tsss-raw.fif" % (subject,
                                                                 condition),
                                                                 preload=True)
raw.plot(highpass=0.1, lowpass=40)

# %%
raw.save(maxfiltered_folder + "%s_%s_mc_tsss-raw.fif" % (subject, condition),
         overwrite=True)

# %%
condition = "interupt"
raw = mne.io.Raw(maxfiltered_folder + "%s_%s_mc_tsss-raw.fif" % (subject,
                                                                 condition),
                                                                 preload=True)
raw.plot(highpass=0.1, lowpass=40)

# %%
raw.save(maxfiltered_folder + "%s_%s_mc_tsss-raw.fif" % (subject, condition),
         overwrite=True)
        