import mne
import sys

from my_settings import (epochs_folder, conditions, mne_folder)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

reject = dict(
    grad=4000e-13,  # T / m (gradiometers)
    mag=4e-12)  # T (magnetometers)

subject = sys.argv[1]

for condition in conditions[:2]:
    epochs = mne.read_epochs(epochs_folder + "%s_%s_ar-epo.fif" % (subject,
                                                                   condition))
    epochs.drop_bad(reject)

    # Make noise cov
    cov = mne.compute_covariance(
        epochs,
        method=['empirical', 'shrunk'],
        tmin=None,
        tmax=-3.5,
        return_estimators=True,
        verbose=True)

    evoked = epochs.average()
    fig = evoked.plot_white(cov, show=False)
    fig.suptitle("subject: %s" % subject)
    fig.savefig(mne_folder + "plots_cov/sub_%s_%s_ar-cov.png" % (subject,
                                                                 condition))
