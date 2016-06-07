import sys
import mne
from mne.minimum_norm import make_inverse_operator, write_inverse_operator

from my_settings import *

subject = sys.argv[1]

for condition in conditions:
    fwd = mne.read_forward_solution(mne_folder + "%s_%s-fwd.fif" %
                                    (subject, condition),
                                    surf_ori=True)
    evoked = mne.read_evokeds(epochs_folder + "%s_%s-ave.fif" % (subject,
                                                                 condition))[0]
    cov = mne.read_cov(mne_folder + "%s_%s-cov.fif" % (subject, condition))

    # make an MEG inverse operator
    inverse_operator = make_inverse_operator(evoked.info,
                                             fwd,
                                             cov,
                                             loose=0.2,
                                             depth=0.8)

    write_inverse_operator(mne_folder + "%s_%s-inv.fif" % (subject, condition),
                           inverse_operator)
