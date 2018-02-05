import numpy as np
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectPercentile, f_classif

from mne.decoding import (SlidingEstimator, cross_val_multiscore, LinearModel)

from my_settings import beamformer_mvpa, bands

for band in bands[1:]:
    X = np.load(beamformer_mvpa + "X_cls_v_pln_%s.npy" % (band))
    y = np.load(beamformer_mvpa + "y_cls_v_pln.npy")

    cv = StratifiedKFold(n_splits=5, shuffle=True)

    clf = make_pipeline(
        StandardScaler(),  # z-score normalization
        SelectPercentile(f_classif,
                         percentile=20),  # select features for speed
        LinearModel(LogisticRegression(C=1)))
    time_decod = SlidingEstimator(clf, n_jobs=2, scoring='roc_auc')

    time_decod.fit(X, y)
    joblib.dump(
        time_decod,
        beamformer_mvpa + "source_cls_v_pln_itc_evk_logreg_%s_FS.jbl" %
        (band))

    scores = cross_val_multiscore(time_decod, X, y, cv=cv)
    np.save(beamformer_mvpa + "source_cls_v_pln_itc_evk_logreg_%s_FS.npy" %
            (band), scores)
