import numpy as np
import mne

from my_settings import (source_folder, subjects_dir)

import matplotlib
matplotlib.use('Agg')

times = np.arange(-4000, 1001, 1)
times = times / 1000.

subjects = [
    "0008", "0009", "0010", "0012", "0014", "0015", "0016", "0017", "0018",
    "0019", "0020", "0021", "0022"
]
bands = ["alpha", "beta", "gamma_low", "gamma_high"]

labels = mne.read_labels_from_annot(
    subject="0008",
    parc="PALS_B12_Brodmann",
    regexp="Brodmann",
    subjects_dir=subjects_dir)

cls_pow_all = np.zeros([len(subjects), len(labels), len(times)])

for j, subject in enumerate(subjects):
    cls = np.load(source_folder + "hilbert_data/%s_classic_ht-epo.npy" %
                  subject).item()

    for band in bands[:1]:
        cls_bs = mne.baseline.rescale(
            np.abs(cls[band])**2, times, baseline=(-3.8, -3.3), mode="zscore")
        cls_pow_all[j, :, :] = cls_bs.mean(axis=0)

        for k, label in enumerate(labels):
            fig = matplotlib.title("condition: cls, label: %s" % (label.name))
            fig = matplotlib.plot(times, cls_pow_all[:, k, :].T)

            fig.savefig(source_folder +
                        "hilbert_data/plots/%s_classic_avg_grad.png" % (
                            label.name, subject))

    # ht_pln = np.load(source_folder + "hilbert_data/%s_plan_ht-epo.npy" %
    #                  subject).item()
    # ht_int = np.load(source_folder + "hilbert_data/%s_interupt_ht-epo.npy" %
    #                  subject).item()
