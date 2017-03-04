import numpy as np
from sklearn.externals import joblib
from mne import create_info, EpochsArray
from mne.decoding import GeneralizationAcrossTime
from sklearn.model_selection import (StratifiedKFold)
from sklearn.metrics import roc_auc_score

from my_settings import (source_folder, data_path)

import matplotlib
matplotlib.use('Agg')

# make time points
window_size = .100  # size in ms of the window to cal measure
step_size = 25  # size in ms of the step to move window
times = np.arange(-4000, 1001, 1)
times = times / 1000.
selected_times = times[::step_size]


# Custom scorer function to use predict_proba
def scorer(y_true, y_pred):
    # Probabilistic estimates are reported for each class. In our case
    # `y_pred` shape is (n_trials, 2), where `y[:, 0] = 1 - y[:, 1]`.
    return roc_auc_score(y_true, y_pred[:, 1])


# Load data
subjects = [
    "0008", "0009", "0010", "0012", "0014", "0015", "0016", "0017", "0018",
    "0019", "0020", "0021", "0022"
]

cls_all = []
pln_all = []

for subject in subjects:
    cls = np.load(source_folder + "graph_data/%s_cls_pow_sliding.npy" %
                  subject).item()

    pln = np.load(source_folder + "graph_data/%s_pln_pow_sliding.npy" %
                  subject).item()

    cls_tmp = []
    cls_tmp.append(cls["cp_alpha"])
    cls_tmp.append(cls["cp_beta"])
    cls_tmp.append(cls["cp_gamma_low"])
    cls_tmp.append(cls["cp_gamma_high"])

    pln_tmp = []
    pln_tmp.append(pln["cp_alpha"])
    pln_tmp.append(pln["cp_beta"])
    pln_tmp.append(pln["cp_gamma_low"])
    pln_tmp.append(pln["cp_gamma_high"])

    cls_all.append(np.asarray(cls_tmp))
    pln_all.append(np.asarray(pln_tmp))

data_cls = np.asarray(cls_all)
data_pln = np.asarray(pln_all)

# Setup data for epochs and cross validation
X = np.vstack([data_cls, data_pln])
y = np.concatenate([np.zeros(len(data_cls)), np.ones(len(data_pln))])
cv = StratifiedKFold(n_splits=7, shuffle=True)

# Create epochs to use for classification
n_trial, n_chan, n_time = X.shape
events = np.vstack((range(n_trial), np.zeros(n_trial, int), y.astype(int))).T
chan_names = ['MEG %i' % chan for chan in range(n_chan)]
chan_types = ['mag'] * n_chan
sfreq = 250
info = create_info(chan_names, sfreq, chan_types)
epochs = EpochsArray(data=X, info=info, events=events, verbose=False)
epochs.times = selected_times[:n_time]

# fit model and score
gat = GeneralizationAcrossTime(
    scorer=scorer, cv=cv, predict_method='predict_proba')
gat.fit(epochs, y=y)
gat.score(epochs, y=y)

# Save model
joblib.dump(gat, data_path + "decode_time_gen/%s_gat_cp.jl" % subject)

# make matrix plot and save it
fig = gat.plot(
    title="Temporal Gen (Classic vs planning): left to right sub: %s" %
    subject)
fig.savefig(data_path + "decode_time_gen/%s_gat_matrix_cp.png" % subject)
