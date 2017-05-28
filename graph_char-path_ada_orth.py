import bct
import numpy as np
import pandas as pd

from my_settings import (source_folder, results_path)

subjects = [
    "0008", "0009", "0010", "0012", "0013", "0014", "0015", "0016",
    "0019", "0020", "0021", "0022"
]

ge_data_all = pd.DataFrame()
lambda_data_all = pd.DataFrame()
dia_data_all = pd.DataFrame()
conditions = ["cls " * 12, "pln " * 12]
conditions = [c.split() for c in conditions]

tois = ["pln", "pre-press", "post-press"]

for toi in tois:
    cls_all = []
    pln_all = []
    for subject in subjects:
        cls = np.load(source_folder + "graph_data/%s_classic_corr_%s_orth.npy" %
                      (subject, toi))

        pln = np.load(source_folder + "graph_data/%s_plan_corr_%s_orth.npy" %
                      (subject, toi))

        cls_all.append(cls.mean(axis=0))
        pln_all.append(pln.mean(axis=0))

    data_cls = [bct.charpath(g) for g in cls_all]
    data_pln = [bct.charpath(g) for g in pln_all]

    # calc global efficiency
    cls_ge = np.asarray([g[1] for g in data_cls])
    pln_ge = np.asarray([g[1] for g in data_pln])

    # calc lambda
    cls_lambda = np.asarray([g[0] for g in data_cls])
    pln_lambda = np.asarray([g[0] for g in data_pln])

    # calc the diameter of the graph
    cls_dia = np.asarray([g[4] for g in data_cls])
    pln_dia = np.asarray([g[4] for g in data_pln])

    ge_data = pd.DataFrame()
    lambda_data = pd.DataFrame()
    dia_data = pd.DataFrame()

    ge_data["ge"] = np.concatenate((cls_ge, pln_ge))
    ge_data["measure"] = "ge"
    ge_data["tio"] = toi
    ge_data["condition"] = conditions

    lambda_data["lambda"] = np.concatenate((cls_lambda, pln_lambda))
    lambda_data["measure"] = "lambda"
    lambda_data["tio"] = toi
    lambda_data["condition"] = conditions

    dia_data["dia"] = np.concatenate((cls_dia, pln_dia))
    dia_data["measure"] = "dia"
    dia_data["tio"] = toi
    dia_data["condition"] = conditions

    ge_data_all = ge_data_all.append(ge_data)
    lambda_data_all = lambda_data_all.append(lambda_data)
    dia_data_all = dia_data_all.append(dia_data)


ge_data_all.to_csv(results_path + "ge_data_all-tois.csv", index=False)
lambda_data_all.to_csv(results_path + "lambda_data_all-tois.csv", index=False)
dia_data_all.to_csv(results_path + "diameter_data_all-tois.csv", index=False)
