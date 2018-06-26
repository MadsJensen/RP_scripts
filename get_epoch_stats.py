import mne
import pandas as pd
import numpy as np

from my_settings import subjects, conditions, erf_raw, misc_folder

data = np.zeros((len(subjects), len(conditions)))

for jj, condition in enumerate(conditions):
    for ii, subject in enumerate(subjects):
        epo = mne.read_epochs(erf_raw + "%s_%s_ar_grads_erf-epo.fif" %
                              (subject[:4], condition))
        data[ii, jj] = len(epo)

df = pd.DataFrame(data=data, columns=list(conditions.keys()))
df.index = subjects

df.to_csv(misc_folder + "epochs_erf_stats.csv")
