import sys
import numpy as np
import mne
from mne.minimum_norm import read_inverse_operator, apply_inverse_epochs

from my_settings import (mne_folder, epochs_folder, source_folder, conditions)

subject = sys.argv[1]

method = "dSPM"
snr = 1.
lambda2 = 1. / snr**2

labels = mne.read_labels_from_annot(
    subject=subject, parc="PALS_B12_Brodmann", regexp="Brodmann")

condition = "classic"

inv = read_inverse_operator(mne_folder + "%s_%s-inv.fif" % (subject,
                                                            condition))
epochs = mne.read_epochs(epochs_folder + "%s_%s-epo.fif" % (subject,
                                                            condition))
# epochs.resample(500)

stcs = apply_inverse_epochs(
    epochs["press"], inv, lambda2, method=method, pick_ori=None)
ts = [
    mne.extract_label_time_course(
        stc, labels, inv["src"], mode="mean_flip") for stc in stcs
]

# for h, tc in enumerate(ts):
#     for j, t in enumerate(tc):
#         t *= np.sign(t[np.argmax(np.abs(t))])
#         tc[j, :] = t
#     ts[h] = tc

ts = np.asarray(ts)
stc.save(source_folder + "%s_%s_epo" % (subject, condition))
np.save(source_folder + "ave_ts/%s_%s_ts-epo.npy" % (subject, condition),
        ts)
