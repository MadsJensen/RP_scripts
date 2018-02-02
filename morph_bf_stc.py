import mne
import sys
from glob import glob

from my_settings import beamformer_results, subjects_dir

subject = sys.argv[1]

file_list = glob(beamformer_results + "%s_*_cor-lh.stc" % subject[:4])

vertices_to = mne.grade_to_vertices('fsaverage', grade=4)


# Compute morph matrix
stc = mne.read_source_estimate(file_list[0][:-7], subject=subject[:4])
morph_mat = mne.compute_morph_matrix(subject[:4], 'fsaverage',
                                     vertices_from=stc.vertices,
                                     vertices_to=vertices_to,
                                     smooth=None,
                                     subjects_dir=subjects_dir)

for ff in file_list:
    stc = mne.read_source_estimate(ff[:-7], subject=subject[:4])
    stc_avg = stc.morph_precomputed('fsaverage', vertices_to, morph_mat,
                                    subject_from=subject[:4])
    stc_avg.save(ff[:-7] + "_avg")
