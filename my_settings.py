import socket

# Setup paths and prepare raw data
hostname = socket.gethostname()

if hostname == "wintermute":
    data_path = "/home/mje/mnt/hyades/scratch4/" +\
                "MINDLAB2015_MEG-CorticalAlphaAttention/"
else:
    data_path = "/projects/MINDLAB2011_24-MEG-readiness/scratch" \
                "/mne_analysis_new"

subjects_dir = "/projects/MINDLAB2011_24-MEG-readiness/scratch/mri/"
save_folder = data_path + "filter_ica_data/"
maxfiltered_folder = data_path + "maxfiltered_data/"
epochs_folder = data_path + "epoched_data/"
tf_folder = data_path + "tf_data/"
mne_folder = data_path + "minimum_norm/"
log_folder = data_path + "log_files/"

