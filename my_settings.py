"""
These are general settings to be used in the current project.

@author: mje
@email: mads [] cnru.dk
"""
import pandas

# Base paths
data_path = "/projects/MINDLAB2011_24-MEG-readiness/scratch/"
misc_folder = "/projects/MINDLAB2011_24-MEG-readiness/misc/"
scripts_folder = "/projects/MINDLAB2011_24-MEG-readiness/scripts/"
subjects_dir = data_path + "fs_subjects_dir/"
trans_dir = data_path + "trans/"
results_folder = "/projects/MINDLAB2011_24-MEG-readiness/result/"

# Beamformer folders
beamformer_raw = data_path + "bf_raw/"
beamformer_filters = data_path + "bf_filters/"
beamformer_source = data_path + "bf_source/"
beamformer_results = data_path + "bf_results/"
beamformer_mvpa = data_path + "bf_mvpa/"

# Erf folders
erf_raw = data_path + "erf_raw/"
erf_filters = data_path + "erf_filters/"
erf_source = data_path + "erf_source/"
erf_results = data_path + "erf_results/"
erf_mvpa = data_path + "erf_mvpa/"

dict_learning = data_path + "dict_learning/"
# Misc
reject_params = dict(
    grad=4000e-13,  # T / m (gradiometers)
    mag=4e-12,  # T (magnetometers)
    eeg=180e-6)

bands = ["Alpha", "Beta", "Gamma_low", "Gamma_high_1", "Gamma_high_2"]
conditions = {"classic": 2, "planning": 4, "interupt": 3}

subjects = [
    "0008", "0009", "0010", "0012", "0013", "0014", "0015", "0016", "0017",
    "0018", "0019", "0020", "0021", "0022"
]

# Bands
iter_freqs = [('Alpha', 8, 12), ('Beta', 13, 30), ('Gamma_low', 30, 45),
              ('Gamma_high_1', 55, 70), ('Gamma_high_2', 70, 90)]

# iter_freqs = [('Alpha', 8, 12), ('Beta', 13, 25), ('Gamma_low', 25, 40),
#               ('Gamma_high_1', 60, 75), ('Gamma_high_2', 75, 90)]


# Functions
def make_rolling_mean(evk, windows_size=20):
    df = evk.to_data_frame()
    return df.rolling(window=windows_size, center=True).mean().values.T


def make_rolling_mean_stc(stc, windows_size=5):
    df = stc.to_data_frame()
    df = df.iloc[:, 1:]
    return df.rolling(window=windows_size, center=True).mean().values.T
