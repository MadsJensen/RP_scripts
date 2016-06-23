import numpy as np
import bct
from sklearn.externals import joblib
from my_settings import *

from sklearn.ensemble import AdaBoostClassifier
from sklearn.cross_validation import (StratifiedShuffleSplit, cross_val_score,
                                      permutation_test_score)
from sklearn.grid_search import GridSearchCV

subjects = ["0008", "0009", "0010", "0012", "0014", "0015", "0016", "0017",
            "0018", "0019", "0020", "0021", "0022"]

cls_all = []
pln_all = []

scores_all = np.empty([4, 10])

results_all = {}

for subject in subjects:
    cls = np.load(source_folder + "graph_data/%s_classic_pow_pre.npy" %
                  subject).item()

    pln = np.load(source_folder + "graph_data/%s_plan_pow_pre.npy" %
                  subject).item()

    cls_all.append(cls)
    pln_all.append(pln)

for k, band in enumerate(bands.keys()):
    data_cls = []
    for j in range(len(cls_all)):
        tmp = cls_all[j][band]
        data_cls.append(np.asarray([bct.centrality.pagerank_centrality(
            g, d=0.85) for g in tmp]).mean(axis=0))
    data_pln = []
    for j in range(len(pln_all)):
        tmp = pln_all[j][band]
        data_pln.append(np.asarray([bct.centrality.pagerank_centrality(
            g, d=0.85) for g in tmp]).mean(axis=0))

    data_cls = np.asarray(data_cls)
    data_pln = np.asarray(data_pln)

    X = np.vstack([data_cls, data_pln])
    y = np.concatenate([np.zeros(len(data_cls)), np.ones(len(data_pln))])

    cv = StratifiedShuffleSplit(y, test_size=0.1)

    model = joblib.load(source_folder +
                        "graph_data/sk_models/pagerank_ada_pre_%s.plk" % band)

    score, perm_scores, pval = permutation_test_score(model,
                                                      X,
                                                      y,
                                                      cv=cv,
                                                      n_permutations=2000,
                                                      n_jobs=4)

    result = {"score": score, "perm_scores": perm_scores, "pval": pval}
    results_all[band] = result

np.save(source_folder + "graph_data/perm_test_pagerank_post.npy", results_all)
