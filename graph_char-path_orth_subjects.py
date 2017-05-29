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
conditions = ['classic', "plan"]

tois = ["pln", "pre-press", "post-press"]

for subject in subjects:
    print("Working on subject: %s" % subject)
    for toi in tois:
        for condition in conditions:
            data_all = []
            pln_all = []
            # noinspection PyAssignmentToLoopOrWithParameter
            data = np.load(source_folder +
                           "graph_data/%s_%s_corr_%s_orth.npy" %
                           (subject, condition, toi))

            graph_data = [bct.charpath(g) for g in data]

            # calc global efficiency
            data_ge = np.asarray([g[1] for g in graph_data])

            # calc lambda
            data_lambda = np.asarray([g[0] for g in graph_data])

            # calc the diameter of the graph
            data_dia = np.asarray([g[4] for g in graph_data])

            ge_data = pd.DataFrame()
            lambda_data = pd.DataFrame()
            dia_data = pd.DataFrame()

            ge_data["ge"] = data_ge
            ge_data["measure"] = "ge"
            ge_data["tio"] = toi
            ge_data["condition"] = condition
            ge_data["subject"] = subject

            lambda_data["lambda"] = data_lambda
            lambda_data["measure"] = "lambda"
            lambda_data["tio"] = toi
            lambda_data["condition"] = condition
            lambda_data["subject"] = subject

            dia_data["dia"] = data_dia
            dia_data["measure"] = "dia"
            dia_data["tio"] = toi
            dia_data["condition"] = condition
            dia_data["subject"] = subject

            ge_data_all = ge_data_all.append(ge_data)
            lambda_data_all = lambda_data_all.append(lambda_data)
            dia_data_all = dia_data_all.append(dia_data)

ge_data_all.to_csv(results_path + "ge_data_all-tois.csv", index=False)
lambda_data_all.to_csv(results_path + "lambda_data_all-tois.csv", index=False)
dia_data_all.to_csv(results_path + "diameter_data_all-tois.csv", index=False)
