import sys

import mne
import numpy as np
from nitime import TimeSeries
from nitime.analysis import CorrelationAnalyzer
from scipy.io import loadmat

from my_settings import (source_folder)

subject = sys.argv[1]

times = np.arange(-4000, 1001, 1)
times = times / 1000.

ht_cls = loadmat(source_folder +
                 "ave_ts/mat_files/%s_classic_ts_DKT_snr-3_orth-epo.mat" %
                 subject)["data"]
ht_pln = loadmat(source_folder +
                 "ave_ts/mat_files/%s_plan_ts_DKT_snr-3_orth-epo.mat" %
                 subject)["data"]
ht_int = loadmat(source_folder +
                 "ave_ts/mat_files/%s_interupt-ts_DKT_snr-3_orth-epo.mat" %
                 subject)["data"]

results_cls = {}
results_pln = {}
results_int = {}

tois = {
    "pln": [1250, 1750],
    "pre-press": [3500, 4000],
    "post-press": [4001, 4500]
}

for toi in tois.keys():
    corr_cls = []
    corr_pln = []
    corr_int = []

    ht_cls_bs = mne.baseline.rescale(
        ht_cls,
        times,
        baseline=(-3.8, -3.3),
        mode="mean")

    ht_pln_bs = mne.baseline.rescale(
        ht_pln,
        times,
        baseline=(-3.8, -3.3),
        mode="mean")
    ht_int_bs = mne.baseline.rescale(
        ht_pln,
        times,
        baseline=(-3.8, -3.3),
        mode="mean")

    for ts in ht_cls_bs:
        nits = TimeSeries(
            ts[:, tois[toi][0]:tois[toi][1]],
            sampling_rate=1000)  # epochs_normal.info["sfreq"])

        corr_cls += [CorrelationAnalyzer(nits)]

    for ts in ht_pln_bs:
        nits = TimeSeries(
            ts[:, tois[toi][0]:tois[toi][1]],
            sampling_rate=1000)  # epochs_normal.info["sfreq"])

        corr_pln += [CorrelationAnalyzer(nits)]

    for ts in ht_int_bs:
        nits = TimeSeries(
            ts[:, tois[toi][0]:tois[toi][1]],
            sampling_rate=1000)  # epochs_normal.info["sfreq"])

        corr_int += [CorrelationAnalyzer(nits)]

    results_cls = np.asarray([c.corrcoef for c in corr_cls])
    results_pln = np.asarray([c.corrcoef for c in corr_pln])
    results_int = np.asarray([c.corrcoef for c in corr_int])

    np.save(source_folder + "graph_data/%s_classic_corr_%s_orth.npy" %
            (subject, toi), results_cls)
    np.save(source_folder + "graph_data/%s_plan_corr_%s_orth.npy" %
            (subject, toi), results_pln)
    np.save(source_folder + "graph_data/%s_interupt_corr_%s_orth.npy" %
            (subject, toi), results_int)
