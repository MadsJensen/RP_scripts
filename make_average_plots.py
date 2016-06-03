# -*- coding: utf-8 -*-
"""
Created on Fri May 27 15:53:42 2016

@author: mje
"""

import mne
import sys

from my_settings import *
import matplotlib
matplotlib.use('Agg')


subject = sys.argv[1]

epo_cls = mne.read_epochs(epochs_folder + "%s_classic-epo.fif" % subject)
epo_pln = mne.read_epochs(epochs_folder + "%s_plan-epo.fif" % subject)
epo_int = mne.read_epochs(epochs_folder + "%s_interupt-epo.fif" % subject)

avg_cls = epo_cls["press"].average()
avg_pln = epo_pln["press"].average()
avg_int_press = epo_int["press"].average()
avg_int_cued = epo_int["cued_press"].average()

avg_cls.save(epochs_folder + "%s_classic-ave.fif" % subject)
avg_pln.save(epochs_folder + "%s_plan-ave.fif" % subject)
avg_int_press.save(epochs_folder + "%s_interupt-ave.fif" % subject)
avg_int_cued.save(epochs_folder + "%s_interupt_cued-ave.fif" % subject)


fig = avg_cls.plot_joint(title="classic")
fig[0].savefig(epochs_folder + "pics/%s_classic_avg_grad.png" % subject)
fig[1].savefig(epochs_folder + "pics/%s_classic_avg_mag.png" % subject)

fig = avg_pln.plot_joint(title="plan")
fig[0].savefig(epochs_folder + "pics/%s_plan_avg_grad.png" % subject)
fig[1].savefig(epochs_folder + "pics/%s_plan_avg_mag.png" % subject)

fig = avg_int_press.plot_joint(title="interupt: press")
fig[0].savefig(epochs_folder + "pics/%s_interupt_avg_grad.png" % subject)
fig[1].savefig(epochs_folder + "pics/%s_interupt_avg_mag.png" % subject)

fig = avg_int_cued.plot_joint(title="interupt: cued")
fig[0].savefig(epochs_folder + "pics/%s_interupt_cued_avg_grad.png" % subject)
fig[1].savefig(epochs_folder + "pics/%s_interupt_cued_avg_mag.png" % subject)
