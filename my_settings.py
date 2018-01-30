"""
These are general settings to be used in the current project.

@author: mje
@email: mads [] cnru.dk
"""
# Base paths
data_path = "/projects/MINDLAB2011_24-MEG-readiness/scratch"
misc_folder = "/projects/MINDLAB2011_24-MEG-readiness/misc"
scripts_folder = "/projects/MINDLAB2011_24-MEG-readiness/scripts"
subjects_dir = data_path + "fs_subjects_dir/"
trans_dir = data_path + "trans/"

# Beamformer folders
beamformer_raw = data_path + "bf_raw/"
beamformer_filters = data_path + "bf_filters/"
beamformer_source = data_path + "bf_source/"
beamformer_results = data_path + "bf_results/"
beamformer_mvpa = data_path + "bf_mvpa/"


# Misc
reject_params = dict(
    grad=4000e-13,  # T / m (gradiometers)
    mag=4e-12,  # T (magnetometers)
    eeg=180e-6)

bands = ["Alpha", "Beta", "Gamma_low", "Gamma_high_1", "Gamma_high_2"]
conditions = {"classic": 2, "planning": 4, "interupt": 3}

# Functions
def get_subjects():
    subjects = list(
        pandas.read_csv("subs_long.txt", header=None).iloc[:, 0].get_values())
    subjects = [s[:4] for s in subjects]
    subjects.remove("0012")
    subjects.remove("0013")
    subjects.remove("0019")

    return subjects


# Bands
iter_freqs = [('Alpha', 8, 12), ('Beta', 13, 30), ('Gamma_low', 30, 45),
              ('Gamma_high_1', 55, 70), ('Gamma_high_2', 70, 90)]
