import sys
import mne
from mne.minimum_norm import make_inverse_operator, write_inverse_operator

from my_settings import (mne_folder, conditions, ica_folder)

subject = sys.argv[1]

for condition in conditions:
    fwd = mne.read_forward_solution(
        mne_folder + "%s_%s_ar-fwd.fif" % (subject, condition), surf_ori=True)
    epochs = mne.read_epochs(ica_folder + "%s_%s_ar_ica-epo.fif" % (
        subject, condition))
    cov = mne.read_cov(mne_folder + "%s_%s_ar-cov.fif" % (subject, condition))

    evoked = epochs.average()
    # make an MEG inverse operator
    inverse_operator = make_inverse_operator(
        evoked.info, fwd, cov, loose=0.2, depth=0.8)

    write_inverse_operator(mne_folder + "%s_%s_ar-inv.fif" %
                           (subject, condition), inverse_operator)
