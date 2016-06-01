import mne
import sys

from my_settings import *

subject = sys.argv[1]
trans = mne_folder + "%s-trans.fif" % subject
bem = subjects_dir + "%s/bem/%s-8192-8192-8182-bem-sol.fif" % (subject,
                                                               subject)
src = mne_folder + "%s-oct-6-src.fif" % subject


for condition in conditions:
    raw_fname = maxfiltered_folder + "%s_%s_mc_tsss-raw.fif" % (subject,
                                                                condition)

    fwd = mne.make_forward_solution(raw_fname, trans=trans, src=src, bem=bem,
                                    fname=None, meg=True, eeg=False,
                                    mindist=5.0, n_jobs=1)
    mne.write_forward_solution(mne_folder + "%s_%s-fwd.fif" % (subject,
                                                               condition),
                               fwd, overwrite=True)
