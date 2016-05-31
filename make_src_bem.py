from __future__ import print_function
import mne
import subprocess

from my_settings import * 

subject = sys.argv[1]

cmd = "/usr/local/common/meeg-cfin/configurations/bin/submit_to_isis"

# make source space
src = mne.setup_source_space(subject, spacing='oct6',
                             subjects_dir=subjects_dir,
                             add_dist=False, overwrite=True)
# save source space
mne.write_source_spaces(mne_folder + "%s-oct-6-src.fif" % subject, src)


setup_forward = "mne_setup_forward_model --subject %s --surf --ico -6" % (
    subject)
subprocess.call([cmd, "1", setup_forward])


# conductivity = (0.3, 0.006, 0.3)  # for three layers
# model = mne.make_bem_model(subject=subject, ico=None,
#                            conductivity=conductivity,
#                            subjects_dir=subjects_dir)
# bem = mne.make_bem_solution(model)

# mne.write_bem_solution(mne_folder + "%s-8194-bem-sol.fif" % subject)
