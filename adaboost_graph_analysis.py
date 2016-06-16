import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.cross_validtion import StratifiedKFold, LeaveOneOut
from sklearn.grid_search import GridSearchCV
# from sklearn.pipeline import make_pipeline

from my_settings import *

subjects = ["0008", "0009", "0010", "0012", "0013", "0014", "0015", "0016",
            "0017", "0018", "0019", "0020", "0021", "0022"]

results_cls = []
results_pln = []

for subject in subjects:
    cls = np.load(source_folder + "graph_data/%s_classic_pow_pln.npy" %
                  subject)
    pln = np.load(source_folder + "graph_data/%s_plan_pow_pln.npy" % subject)

    results_cls.append(cls.mean(axis=0))
    results_pln.append(pln.mean(axis=0))

X = np.vstack([results_cls, results_pln])
y = np.concatenate([np.zeros(len(results_cls)), np.ones(len(results_pln))])

cv = StratifiedKFold(y, n_folds=10)
llo = LeaveOneOut(y)

ada = AdaBoostClassifier()

adaboost_params = {"n_estimators": np.arange(10, 1000, 100)}

ada_grid = GridSearchCV(ada, param_grid=adaboost_params, scoring="roc_auc")
