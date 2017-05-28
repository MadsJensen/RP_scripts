import socket

# Setup paths and prepare raw data
hostname = socket.gethostname()

reject_params = dict(
    grad=4000e-13,  # T / m (gradiometers)
    mag=4e-12  # T (magnetometers)
)

conditions = ["classic", "plan", "interupt"]

bands = {
    "alpha": [8, 12],
    "beta": [13, 25],
    "gamma_low": [30, 48],
    "gamma_high": [52, 90]
}

window_size = .200  # size in ms of the window to cal measure
step_size = 50  # size in ms of the step to move window

data_path = "/projects/MINDLAB2011_24-MEG-readiness/scratch" \
                "/mne_analysis_new/"
subjects_dir = "/projects/MINDLAB2011_24-MEG-readiness/scratch/" + \
               "fs_subjects_dir/"
save_folder = data_path + "filter_ica_data/"
maxfiltered_folder = data_path + "maxfiltered_data/"
mf_autobad_off_folder = data_path + "maxfiltered_data_autobad-off/"
epochs_folder = data_path + "epoched_data/"
ica_folder = data_path + "ica_data/"
tf_folder = data_path + "tf_data/"
mne_folder = data_path + "minimum_norm/"
log_folder = data_path + "log_files/"
source_folder = data_path + "source_data/"
compare_folder = data_path + "compare_src_epo-ave"
results_path = "/projects/MINDLAB2011_24-MEG-readiness/result/"
