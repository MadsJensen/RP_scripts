# -*- coding: utf-8 -*-
"""
Created on Fri May 27 15:53:42 2016

@author: mje
"""

import mne
import sys

import matplotlib
matplotlib.use('Agg')

from my_settings import *

subject = sys.argv[1]

epo_cls = mne.read_epochs(epochs_folder + "%s_classic-epo.fif" % subject)
epo_pln = mne.read_epochs(epochs_folder + "%s_plan-epo.fif" % subject)
epo_int = mne.read_epochs(epochs_folder + "%s_interupt-epo.fif" % subject)

avg_cls = epo_cls.average()
avg_pln = epo_pln.average()
avg_int = epo_int.average()


fig = avg_cls.plot_joint(title="classic")
fig[0].savefig(epochs_folder + "pics/%s_classic_avg_grad.png" % subject)
fig[1].savefig(epochs_folder + "pics/%s_classic_avg_mag.png" % subject)

avg_pln.plot_joint(title="plan")
fig[0].savefig(epochs_folder + "pics/%s_plan_avg_grad.png" % subject)
fig[1].savefig(epochs_folder + "pics/%s_plan_avg_mag.png" % subject)

avg_int.plot_joint(title="interupt")
fig[0].savefig(epochs_folder + "pics/%s_interupt_avg_grad.png" % subject)
fig[1].savefig(epochs_folder + "pics/%s_interupt_avg_mag.png" % subject)
