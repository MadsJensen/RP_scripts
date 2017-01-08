import sys
import numpy as np
import mne
from mne.minimum_norm import read_inverse_operator, apply_inverse

from my_settings import (mne_folder, epochs_folder, conditions, source_folder)

subject = sys.argv[1]

method = "dSPM"
snr = 3.
lambda2 = 1. / snr**2

labels = mne.read_labels_from_annot(
    subject=subject, parc="PALS_B12_Brodmann", regexp="Brodmann")

for condition in conditions:
    inv = read_inverse_operator(mne_folder + "%s_%s-inv.fif" % (subject,
                                                                condition))
    evoked = mne.read_evokeds(epochs_folder + "%s_%s-ave.fif" % (subject,
                                                                 condition))[0]

    stc = apply_inverse(evoked, inv, lambda2, method=method, pick_ori=None)
    ts = mne.extract_label_time_course(
        stc, labels, inv["src"], mode="mean_flip")
    # for j, t in enumerate(ts):
    #     t *= np.sign(t[np.argmax(np.abs(t))])
    #     ts[j, :] = t

    stc.save(source_folder + "%s_%s_ave" % (subject, condition))
    np.save(source_folder + "ave_ts/%s_%s_ts.npy" % (subject, condition), ts)
