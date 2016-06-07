"""Epoch a raw data set.

"""
import mne
import sys
import matplotlib
matplotlib.use('Agg')

from my_settings import *

subject = sys.argv[1]

# SETTINGS

for condition in conditions:
    epochs = mne.read_epochs(epochs_folder + "%s_%s-epo.fif" % (subject,
                                                                condition))

    cov = mne.compute_covariance(epochs["press"], tmax=-3.5,
                                     method="factor_analysis")

    cov.save(mne_folder + "%s_%s-cov.fif" % (subject, condition))


    evoked = epochs["press"].average()
    fig = evoked.plot_white(cov)
    fig.savefig(mne_folder + "plots_cov/%s_%s_cov.png" % (subject, condition))
