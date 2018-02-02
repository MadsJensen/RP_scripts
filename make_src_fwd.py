import mne
import sys
from my_settings import (beamformer_raw, beamformer_source, subjects_dir,
                         trans_dir, conditions)

subject = sys.argv[1]

trans_fname = trans_dir + "%s_planning-trans.fif" % subject[:4]
bem = subjects_dir + "%s/bem/%s-8192-8192-8192-bem-sol.fif" % (subject[:4],
                                                               subject[:4])
# t1_fname = op.join(subjects_dir, subject, 'mri/T1.mgz')  # T1 MRI

# set up a volume source space
# src = mne.setup_volume_source_space(subject, pos=5., mri=t1_fname, bem=bem)
# mne.write_source_spaces(beamformer_source + "%s_vol-src.fif" % subject[:4],
#                         src)

src = subjects_dir + "%s/bem/%s-oct-6-src.fif" % (subject[:4], subject[:4])

# make leadfield
for condition in conditions.keys():
    epochs = mne.read_epochs(
        beamformer_raw + "%s_%s_Alpha_ar_grads-epo.fif" %
        (subject[:4], condition),
        preload=False)

    fwd = mne.make_forward_solution(
        epochs.info,
        trans=trans_fname,
        src=src,
        bem=bem,
        meg=True,
        eeg=False,
        n_jobs=1)

    # let's save it to disk:
    mne.write_forward_solution(
        beamformer_source + "%s_%s_cor-fwd.fif" % (subject[:4], condition),
        fwd,
        overwrite=True)
 