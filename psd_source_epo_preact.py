import mne
from mne.minimum_norm import read_inverse_operator, compute_source_psd_epochs
import sys
import cPickle as pickle

from my_settings import (mne_folder, conditions, source_folder, epochs_folder)

subject = sys.argv[1]

labels = mne.read_labels_from_annot(
    subject=subject, parc="PALS_B12_Brodmann", regexp="Brodmann")

snr = 1.0  # use smaller SNR for epo data
lambda2 = 1.0 / snr**2
method = "dSPM"  # use dSPM method (could also be MNE or sLORETA)

for condition in conditions:

    # Load data
    inv = read_inverse_operator(mne_folder + "%s_%s-inv.fif" % (subject,
                                                                condition))
    epochs = mne.read_epochs(epochs_folder + "%s_%s-epo.fif" % (subject,
                                                                condition))
    # define frequencies of interest
    fmin, fmax = 0., 90.
    bandwidth = 2.  # bandwidth of the windows in Hz

    # compute source space psd in label

    # Note: By using "return_generator=True" stcs will be a generator object
    # instead of a list. This allows us so to iterate without having to
    # keep everything in memory.
    epochs.crop(-0.5, 0)
    stcs = compute_source_psd_epochs(
        epochs,
        inv,
        lambda2=lambda2,
        method=method,
        fmin=fmin,
        fmax=fmax,
        bandwidth=bandwidth,
        return_generator=False)

    pickle.dump(stcs, open(source_folder +
                "%s_%s_psd_preact_epo.pkl" % (subject, condition), 'wb'))
