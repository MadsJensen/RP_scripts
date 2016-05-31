from __future__ import print_function
import mne

from my_settings import * 

subject = sys.argv[1]

# make source space
src = mne.setup_source_space(subject, spacing='oct6',
                             subjects_dir=subjects_dir,
                             add_dist=False, overwrite=True)
# save source space
mne.write_source_spaces(mne_folder + "%s-oct6-src.fif" % subject, src)

conductivity = (0.3, 0.006, 0.3)  # for three layers
model = mne.make_bem_model(subject=subject, ico=None,
                           conductivity=conductivity,
                           subjects_dir=subjects_dir)
bem = mne.make_bem_solution(model)

mne.write_bem_solution(mne_folder + "%s-8194-bem-sol.fif" % subject)
