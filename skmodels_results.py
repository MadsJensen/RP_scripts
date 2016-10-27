# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 14:50:08 2016

@author: mje
"""

import numpy as np
import pandas as pd
from sklearn.externals import joblib
import mne
import glob 

from my_settings import *

labels = mne.read_labels_from_annot(subject="0008",
                                    parc="PALS_B12_Brodmann",
                                    regexp="Brodmann",
                                    subjects_dir=subjects_dir)

conditions = ["pln", "pre", "post"]
# measures =  ["eigen", "pagerank", "path-strength"]
measures =  ["pagerank"]
bands = ["alpha", "beta", "gamma_low", "gamma_high"]

column_keys = ["condition", "measure", "band", "scores", 
               "mean_score", "std", "feature_importance"]

results = pd.DataFrame(columns=column_keys)

for cond in conditions:
    for measure in measures:
        scores = np.load(source_folder + "graph_data/%s_scores_all_ada_%s.npy" % (measure, cond))
        for j, band in enumerate(bands):  
            model = joblib.load(source_folder +
                      "graph_data/sk_models/%s_ada_%s_%s.plk" % (measure, condition, band))
            row = pd.DataFrame([{"condition": cond,
                                 "measure": measure,
                                 "band": band,
                                 "scores": scores[j],
                                 "mean_score": scores[j].mean(),
                                 "std": scores[j].std(),
                                 "feature_importance": model.feature_importances_}])
            results = results.append(row, ignore_index=True)

results[["condition", "measure", "band", "mean_score", "std"]].sort("mean_score")    

for condit  
for j in range(len(results.feature_importance)):
    print("\nmeasure: %s, band: %s" % (results.ix[j].measure.swapcase(),
                                      results.ix[j].band.swapcase()))
    for i in range(82):
        if results.feature_importance[j][i] > 0:
            print(labels[i].name + "  score: %s" %
                np.round(results.feature_importance[j][i], 4))

for i in range(82):
        if f3[i] > 0:
            print(labels[i].name + "  score: %s" % f3[i])



#### PERM TESTS ####
perm_tests = glob.glob(source_folder + "graph_data/perm_test*")

column_keys = ["name", "pval", "score", "mean_perm", "std_perm", "band"]
perm_results = pd.DataFrame(columns=column_keys)

for test in perm_tests:
    tmp = np.load(test).item()
    for band in bands:
        name = test.split("_")[-2:]
        name = "_".join(name)[:-4]
        row = pd.DataFrame([{"name": name,
                             "pval": tmp[band]["pval"],
                             "score": tmp[band]["score"],
                             "mean_perm": tmp[band]["perm_scores"].mean(),
                             "std_perm":tmp[band]["perm_scores"].std(),
                             "band": band}])
        perm_results = perm_results.append(row, ignore_index=True)
        
        
        
        
        
        
        
                  
