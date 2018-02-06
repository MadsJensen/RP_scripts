import sys

import mne
import numpy as np
import pandas as pd
from my_settings import (beamformer_source, results_folder)

subject = sys.argv[1]

fwd_pln = mne.read_forward_solution(
    beamformer_source + "%s_planning_cor-fwd.fif" % subject[:4])
fwd_cls = mne.read_forward_solution(
    beamformer_source + "%s_classic_cor-fwd.fif" % subject[:4])

data = {"cls_mean": fwd_cls["sol"]["data"].mean(),
        "pln_mean": fwd_pln["sol"]["data"].mean(),
        "cls_std": fwd_cls["sol"]["data"].std(),
        "pln_std": fwd_pln["sol"]["data"].std(),
        "diff_mean": fwd_cls["sol"]["data"].mean() -
        fwd_pln["sol"]["data"].mean(),
        "diff_std": fwd_cls["sol"]["data"].std() -
        fwd_pln["sol"]["data"].std()}

df = pd.DataFrame.from_dict(data, orient="index").T
df.to_csv(results_folder + "%s_fwd_values.csv" % subject[:4], index=False)

fwd_avg = mne.average_forward_solutions([fwd_pln, fwd_cls])
mne.write_forward_solution(beamformer_source + "%s_avg_cor-fwd.fif" %
                           subject[:4],
                           fwd_avg)
