import numpy as np
import bct
from my_settings import *

from permute.core import two_sample

subjects = ["0008", "0009", "0010", "0012", "0014", "0015", "0016",
            "0017", "0018", "0019", "0020", "0021", "0022"]


scores_all = np.empty([4, 10])
cls_all = []
pln_all = []

for subject in subjects:
    cls = np.load(source_folder + "graph_data/%s_classic_pow_pln.npy" %
                  subject).item()

    pln = np.load(source_folder + "graph_data/%s_plan_pow_pln.npy" %
                  subject).item()

    cls_all.append(cls)
    pln_all.append(pln)


results_cls = dict()
results_pln = dict()

for k, band in enumerate(bands.keys()):
    data_cls = []
    for j in range(len(cls_all)):
        tmp = cls_all[j][band]
        data_cls.append(np.asarray([bct.transitivity_wu(g)
        for g in tmp]).mean(axis=0))
    data_pln = []
    for j in range(len(pln_all)):
        tmp = pln_all[j][band]
        data_pln.append(np.asarray([bct.transitivity_wu(g)
        for g in tmp]).mean(axis=0))

    data_cls = np.asarray(data_cls)
    data_pln = np.asarray(data_pln)
    
    results_cls[band] = data_cls
    results_pln[band] = data_pln


results_perm = dict()
for band in bands.keys():
    print(band)
    data = {'cls': results_cls[band],
            'pln':results_pln[band]}

    model = best.make_model( data )

    M = MCMC(model)
    M.sample(iter=110000, burn=10000)
    results_perm[band] = M


fig = best.plot.make_figure(M)
fig.savefig('smart_drug.png',dpi=70)

    p, t = two_sample(results_cls[band], results_pln[band], stat='t',
                      alternative='two-sided')
    results_perm[band] = p                  

