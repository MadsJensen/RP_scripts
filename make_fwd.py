import mne
import sys
import glob

from my_settings import (subjects_dir, ica_folder, mne_folder, conditions)

subject = sys.argv[1]

trans = mne_folder + "%s-trans.fif" % subject
bem = glob.glob(subjects_dir + "%s/bem/%s-*-bem-sol.fif" % (subject, subject))[
    0]
src = mne_folder + "%s-oct-6-src.fif" % subject

for condition in conditions:
    epochs_fname = ica_folder + "%s_%s_ar_ica-epo.fif" % (subject, condition)

    fwd = mne.make_forward_solution(
        epochs_fname,
        trans=trans,
        src=src,
        bem=bem,
        fname=None,
        meg=True,
        eeg=False,
        mindist=5.0,
        n_jobs=1)

    mne.write_forward_solution(
        mne_folder + "%s_%s_ar-fwd.fif" % (subject, condition),
        fwd,
        overwrite=True)
