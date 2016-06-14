import mne
import numpy as np
from mne.stats import permutation_cluster_test

from my_settings import *

subjects = ["0008", "0009", "0010", "0012", "0013", "0014", "0015", "0016",
            "0017", "0018", "0019", "0020", "0021", "0022"]

times = np.arange(-4000, 1001, 1)
times = times / 1000.

cls_all = []
int_all = []

results_all = {}

for band in bands.keys():
    cls_all = []
    int_all = []

    for subject in subjects:
        ht_cls = np.load(source_folder + "hilbert_data/%s_classic_ht-epo.npy" %
                         subject).item()
        ht_int = np.load(source_folder + "hilbert_data/%s_interupt_ht-epo.npy"
                         % subject).item()

        ht_cls_bs = mne.baseline.rescale(
            np.abs(ht_cls[band])**2,
            times,
            baseline=(-3.8, -3.3),
            mode="zscore")
        ht_int_bs = mne.baseline.rescale(
            np.abs(ht_int[band])**2,
            times,
            baseline=(-3.8, -3.3),
            mode="zscore")
        cls_all += [ht_cls_bs.mean(axis=0)]
        int_all += [ht_int_bs.mean(axis=0)]

    cls_all = np.asarray(cls_all)
    int_all = np.asarray(int_all)

    cluster_results = []

    for label_nr, label in enumerate(labels):
        data_1 = cls_all[:, label_nr, :]
        data_2 = int_all[:, label_nr, :]

        # Compute statistic

        T_obs, clusters, cluster_p_values, H0 = \
            permutation_cluster_test([data_1, data_2],
                                     n_permutations=10000, tail=0, n_jobs=1)

        cluster_results += [cluster_p_values]

    results_all[band] = cluster_results

np.save(source_folder + "hilbert_data/perm_test_cls-int_full.npy", results_all)
