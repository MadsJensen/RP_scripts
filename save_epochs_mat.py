import sys
import numpy as np
from scipy.io import savemat

from my_settings import (source_folder, conditions)

subject = sys.argv[1]

for condition in conditions:
    data = np.load(source_folder + "ave_ts/%s_%s_ts_DKT_snr-3-epo.npy" %
                   (subject, condition))
    savemat(source_folder +
            "ave_ts/mat_files/%s_%s_ts_DKT_snr-3-epo" %
            (subject, condition), dict(data=data))
