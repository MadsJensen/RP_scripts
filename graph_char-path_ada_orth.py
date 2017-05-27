import bct
import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.externals import joblib
from sklearn.model_selection import (GridSearchCV, StratifiedKFold,
                                     cross_val_score)

from my_settings import (source_folder)

subjects = [
    "0008", "0009", "0010", "0012", "0013", "0014", "0015", "0016",
    "0019", "0020", "0021", "0022"
]

cls_all = []
pln_all = []
for subject in subjects:
    cls = np.load(source_folder + "graph_data/%s_classic_corr_pln_orth.npy" %
                  subject)

    pln = np.load(source_folder + "graph_data/%s_plan_corr_pln_orth.npy" %
                  subject)

    cls_all.append(cls.mean(axis=0))
    pln_all.append(pln.mean(axis=0))

data_cls = [bct.charpath(g) for g in cls_all]
data_pln = [bct.charpath(g) for g in pln_all]

cls_ge = np.asarray([g[1] for g in data_cls])
pln_ge = np.asarray([g[1] for g in data_pln])

cls_lambda = np.asarray([g[0] for g in data_cls])
pln_lambda = np.asarray([g[0] for g in data_pln])

ge_data = pd.DataFrame()
ge_data["pln"] = pln_ge
ge_data["cls"] = cls_ge


lambda_data = pd.DataFrame()
lambda_data["pln"] = pln_lambda
lambda_data["cls"] = cls_lambda

cls_dia= np.asarray([g[4] for g in data_cls])
pln_dia= np.asarray([g[4] for g in data_pln])

diadata = pd.DataFrame()
diadata["pln"] = pln_dia
diadata["cls"] = cls_dia
