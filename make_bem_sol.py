from __future__ import print_function
import subprocess
import os

from my_settings import *

# subject = sys.argv[1]

cmd = "/usr/local/common/meeg-cfin/configurations/bin/submit_to_isis"
os.environ["SUBJECTS_DIR"] = subjects_dir

subjects = ["0008", "0009", "0010", "0011", "0012", "0013",
            "0014", "0015", "0016", "0017", "0018", "0019", "0020",
            "0021", "0022"]

for subject in subjects:
    setup_forward = "mne_setup_forward_model --subject %s --surf --ico -6" % (
        subject)
    subprocess.call([cmd, "2", setup_forward])
