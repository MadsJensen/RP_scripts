"""Calculate noice covariance matrix from baseline.

"""
import mne
import sys

from my_settings import (conditions, mne_folder, epochs_folder)

import matplotlib
matplotlib.use('Agg')

subject = sys.argv[1]

for condition in conditions:
    epochs = mne.read_epochs(epochs_folder + "%s_%s_ar-epo.fif" % (subject,
                                                                   condition))

    cov = mne.compute_covariance(
        epochs["press"], tmin=None, tmax=-3.5, method="shrunk")

    cov.save(mne_folder + "%s_%s_ar-cov.fif" % (subject, condition))

    evoked = epochs["press"].average()
    fig = evoked.plot_white(cov)
    fig.savefig(mne_folder + "plots_cov/%s_%s_ar_cov.png" % (subject, condition
                                                             ))
