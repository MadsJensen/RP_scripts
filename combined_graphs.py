import mne
import sys
import numpy as np

from my_settings import *

subjects = ["0008", "0009", "0010", "0012", "0013", "0014", "0015", "0016",
            "0017", "0018", "0019", "0020", "0021", "0022"]

results_cls = []
results_pln = []

for subject in subjects:
    cls = np.load(source_folder + "graph_data/%s_classic_pow_pln.npy" %
                  subject)
    pln = np.load(source_folder + "graph_data/%s_plan_pow_pln.npy" %
                  subject)

    results_cls.append(cls.mean(axis=0))
    results_pln.append(pln.mean(axis=0))


    
