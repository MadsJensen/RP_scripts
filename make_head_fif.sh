
export SUBJECTS_DIR = "/projects/MINDLAB2011_24-MEG-readiness/scratch/fs_subjects_dir"

for i in $(seq -f "%04g" 8 22)
do
    mne_surf2bem --surf $SUBJECTS_DIR/$1/bem/outer_skin.surf --fif $SUBJECTS_DIR/$1/bem/$-head.fif --id 4
done
