# -*- coding: utf-8 -*-
"""
Created on Fri May 27 15:53:42 2016

@author: mje
"""

import mne

from my_settings import *

subject = "0021"

epo_cls = mne.read_epochs(epochs_folder + "%s_classic-epo.fif" % subject)
epo_pln = mne.read_epochs(epochs_folder + "%s_plan-epo.fif" % subject)
epo_int = mne.read_epochs(epochs_folder + "%s_interupt-epo.fif" % subject)

avg_cls = epo_cls.average()
avg_pln = epo_pln.average()
avg_int = epo_int.average()


avg_cls.crop(tmin=-.5, tmax=0).plot_joint(title="classic")
avg_pln.crop(tmin=-.5, tmax=0).plot_joint(title="plan")
avg_int.crop(tmin=-.5, tmax=0).plot_joint(title="interupt")