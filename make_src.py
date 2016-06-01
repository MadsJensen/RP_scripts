from __future__ import print_function
import mne
import subprocess
import sys
import os

from my_settings import * 

subject = sys.argv[1]

# make source space
src = mne.setup_source_space(subject, spacing='oct6',
                             subjects_dir=subjects_dir,
                             add_dist=False, overwrite=True)
# save source space
mne.write_source_spaces(mne_folder + "%s-oct-6-src.fif" % subject, src)
