import numpy as np
import bct
import sys
import mne

from nitime import TimeSeries
from nitime.analysis import CorrelationAnalyzer

from my_settings import (bands, source_folder, window_size, step_size)
subject = sys.argv[1]

cls = np.load(source_folder + "hilbert_data/%s_classic_ht-epo.npy" %
              subject).item()
pln = np.load(source_folder + "hilbert_data/%s_plan_ht-epo.npy" %
              subject).item()

times = np.arange(-4000, 1001, 1)
times = times / 1000.
selected_times = times[::step_size]

results_cls = {}
results_pln = {}

for k, band in enumerate(bands.keys()):
    # baseline correct timeseries
    cls_bs = mne.baseline.rescale(
        np.abs(cls[band])**2, times, baseline=(-3.8, -3.3), mode="zscore")

    pln_bs = mne.baseline.rescale(
        np.abs(pln[band])**2, times, baseline=(-3.8, -3.3), mode="zscore")

    deg_cls = []
    deg_pln = []
    trans_cls = []
    trans_pln = []
    ge_cls = []
    ge_pln = []
    cp_cls = []
    cp_pln = []

    for st in selected_times:
        if st + window_size < times[-1]:

            from_time = np.abs(times - st).argmin()
            to_time = np.abs(times - (st + window_size)).argmin()
            corr_cls = []
            corr_pln = []

            # make timeseries object
            for ts in cls_bs:
                nits = TimeSeries(
                    ts[:, from_time:to_time],
                    sampling_rate=1000)  # epochs_normal.info["sfreq"])

                corr_cls += [CorrelationAnalyzer(nits)]

            for ts in pln_bs:
                nits = TimeSeries(
                    ts[:, from_time:to_time],
                    sampling_rate=1000)  # epochs_normal.info["sfreq"])

                corr_pln += [CorrelationAnalyzer(nits)]

            corr_cls_coef = [d.corrcoef for d in corr_cls]
            corr_pln_coef = [d.corrcoef for d in corr_pln]

            full_matrix = np.concatenate(
                [corr_cls_coef, corr_pln_coef], axis=0)
            threshold = np.median(full_matrix[np.nonzero(full_matrix)]) + \
                np.std(full_matrix[np.nonzero(full_matrix)])

            data_cls_bin = corr_cls_coef > threshold
            data_pln_bin = corr_pln_coef > threshold

            deg_cls_tmp = np.asarray([
                bct.degrees_und(g)
                for g in data_cls_bin
            ]).mean(axis=0)

            deg_pln_tmp = np.asarray([
                bct.degrees_und(g)
                for g in data_pln_bin
            ]).mean(axis=0)

            trans_cls_tmp = np.asarray(
                [bct.transitivity_bu(g) for g in data_cls_bin]).mean(axis=0)
            trans_pln_tmp = np.asarray(
                [bct.transitivity_bu(g) for g in data_pln_bin]).mean(axis=0)

            ge_cls_tmp = np.asarray(
                [bct.distance.charpath(g)[1] for g in data_cls_bin]).mean(
                    axis=0)
            ge_pln_tmp = np.asarray(
                [bct.distance.charpath(g)[1] for g in data_pln_bin]).mean(
                    axis=0)

            cp_cls_tmp = np.asarray(
                [bct.distance.charpath(g)[0] for g in data_cls_bin]).mean(
                    axis=0)
            cp_pln_tmp = np.asarray(
                [bct.distance.charpath(g)[0] for g in data_pln_bin]).mean(
                    axis=0)

            # Add measure to results list
            deg_cls.append(deg_cls_tmp)
            deg_pln.append(deg_pln_tmp)
            trans_cls.append(trans_cls_tmp)
            trans_pln.append(trans_pln_tmp)
            ge_cls.append(ge_cls_tmp)
            ge_pln.append(ge_pln_tmp)
            cp_cls.append(cp_cls_tmp)
            cp_pln.append(cp_pln_tmp)

    results_cls["deg_%s" % band] = np.asarray(deg_cls)
    results_pln["deg_%s" % band] = np.asarray(deg_pln)
    results_cls["trans_%s" % band] = np.asarray(trans_cls)
    results_pln["trans_%s" % band] = np.asarray(trans_pln)
    results_cls["ge_%s" % band] = np.asarray(ge_cls)
    results_pln["ge_%s" % band] = np.asarray(ge_pln)
    results_cls["cp_%s" % band] = np.asarray(cp_cls)
    results_pln["cp_%s" % band] = np.asarray(cp_pln)

np.save(source_folder + "graph_data/%s_pln_pow_sliding_bin.npy" % (subject),
        results_pln)
np.save(source_folder + "graph_data/%s_cls_pow_sliding_bin.npy" % (subject),
        results_cls)
