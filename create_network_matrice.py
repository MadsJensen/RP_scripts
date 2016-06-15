import mne
import sys
import numpy as np
from nitime import TimeSeries
from nitime.analysis import CorrelationAnalyzer

from my_settings import *

subject = sys.argv[1]

times = np.arange(-4000, 1001, 1)
times = times / 1000.

ht_cls = np.load(source_folder + "hilbert_data/%s_classic_ht-epo.npy" %
                 subject).item()
ht_pln = np.load(source_folder + "hilbert_data/%s_plan_ht-epo.npy" %
                 subject).item()
ht_int = np.load(source_folder + "hilbert_data/%s_interupt_ht-epo.npy" %
                 subject).item()

results_cls = {}
results_pln = {}
results_int = {}

for band in bands.keys():
    cls_all = []
    pln_all = []
    int_all = []

    ht_cls_bs = mne.baseline.rescale(
        np.abs(ht_cls[band])**2,
        times,
        baseline=(-3.8, -3.3),
        mode="zscore")
    ht_pln_bs = mne.baseline.rescale(
        np.abs(ht_pln[band])**2,
        times,
        baseline=(-3.8, -3.3),
        mode="zscore")
    ht_int_bs = mne.baseline.rescale(
        np.abs(ht_pln[band])**2,
        times,
        baseline=(-3.8, -3.3),
        mode="zscore")

    cls_all += [ht_cls_bs.mean(axis=0)]
    pln_all += [ht_pln_bs.mean(axis=0)]
    int_all += [ht_int_bs.mean(axis=0)]

    cls_all = np.asarray(cls_all)
    pln_all = np.asarray(pln_all)
    int_all = np.asarray(int_all)

    for ts in ht_cls_bs:
        nits = TimeSeries(ts[:, 1250:1750],
                          sampling_rate=1000)  # epochs_normal.info["sfreq"])

        corr_cls += [CorrelationAnalyzer(nits)]

    for ts in ht_pln_bs:
        nits = TimeSeries(ts[:, 1250:1750],
                          sampling_rate=1000)  # epochs_normal.info["sfreq"])

        corr_pln += [CorrelationAnalyzer(nits)]

    for ts in ht_int_bs:
        nits = TimeSeries(ts[:, 1250:1750],
                          sampling_rate=1000)  # epochs_normal.info["sfreq"])

        corr_int += [CorrelationAnalyzer(nits)]

    corr_cls = np.asarray([c.corrcoef for c in corr_cls])
    corr_pln = np.asarray([c.corrcoef for c in corr_pln])
    corr_int = np.asarray([c.corrcoef for c in corr_int])

    results_cls[band] = corr_cls
    results_pln[band] = corr_pln
    results_int[band] = corr_int

np.save(source_folder + "graph_data/%s_classic_pow_pln.npy" % subject,
        results_cls)
np.save(source_folder + "graph_data/%s_plan_pow_pln.npy" % subject,
        results_pln)
np.save(source_folder + "graph_data/%s_interupt_pow_pln.npy" % subject,
        results_int)
