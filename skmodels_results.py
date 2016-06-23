# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 14:50:08 2016

@author: mje
"""

import numpy as np
import pandas as pd
from sklearn.externals import joblib
import mne

from my_settings import *

labels = mne.read_labels_from_annot(subject="0008",
                                    parc="PALS_B12_Brodmann",
                                    regexp="Brodmann",
                                    subjects_dir=subjects_dir)

measures =  ["pagerank"]
bands = ["alpha", "beta", "gamma_low", "gamma_high"]

column_keys = ["measure", "band", "scores", 
               "mean_score", "std", "feature_importance"]

results = pd.DataFrame(columns=column_keys)

for measure in measures:
    scores = np.load(source_folder + "graph_data/%s_scores_all_xgb.npy" % measure)
    for j, band in enumerate(bands):  
        model = joblib.load(source_folder +
                  "graph_data/sk_models/%s_xgb_%s.plk" % (measure, band))
        row = pd.DataFrame([{"measure": measure,
                             "band": band,
                             "scores": scores[j],
                             "mean_score": scores[j].mean(),
                             "std": scores[j].std(),
                             "feature_importance": model.feature_importances_}])
        results = results.append(row, ignore_index=True)
    
              
for j in range(len(results.feature_importance)):
    print("\nmeasure: %s, band: %s" % (results.ix[j].measure.swapcase(),
                                      results.ix[j].band.swapcase()))
    for i in range(82):
        if results.feature_importance[j][i] > 0:
            print(labels[i].name + "  score: %s" %
                np.round(results.feature_importance[j][i], 4))
                  

sns.barplot(y="mean_score", x="band", hue="measure", data=foo)
