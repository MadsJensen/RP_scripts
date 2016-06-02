import socket

# Setup paths and prepare raw data
hostname = socket.gethostname()

reject_params = dict(grad=4000e-13,  # T / m (gradiometers)
                     mag=4e-12  # T (magnetometers)
                     )

conditions = ["classic", "plan", "interupt"]

data_path = "/projects/MINDLAB2011_24-MEG-readiness/scratch" \
                "/mne_analysis_new/"

subjects_dir = "/projects/MINDLAB2011_24-MEG-readiness/scratch/fs_subjects_dir/"
save_folder = data_path + "filter_ica_data/"
maxfiltered_folder = data_path + "maxfiltered_data/"
epochs_folder = data_path + "epoched_data/"
tf_folder = data_path + "tf_data/"
mne_folder = data_path + "minimum_norm/"
log_folder = data_path + "log_files/"
source_folder = data_path + "source_data"
