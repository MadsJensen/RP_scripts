import numpy as np
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression, LassoCV
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import (SelectFromModel, SelectPercentile,
                                       f_classif)

from mne.decoding import (SlidingEstimator, cross_val_multiscore, LinearModel)

from my_settings import beamformer_mvpa, bands

n_jobs = 4

for band in bands:
    X = np.load(beamformer_mvpa + "X_cls_v_pln_%s.npy" % (band))
    y = np.load(beamformer_mvpa + "y_cls_v_pln.npy")

    cv = StratifiedKFold(n_splits=5, shuffle=True)

    clf = make_pipeline(
        StandardScaler(),  # z-score normalization
        # SelectPercentile(f_classif, percentile=40),
        SelectFromModel(
            LassoCV(
                cv=StratifiedKFold(n_splits=5, shuffle=True),
                max_iter=5000,
                n_jobs=n_jobs),
            threshold="0.1*mean"),
        LinearModel(LogisticRegression(C=1)))

    time_decod = SlidingEstimator(clf, n_jobs=2, scoring='roc_auc')

    time_decod.fit(X, y)
    joblib.dump(
        time_decod,
        beamformer_mvpa + "source_cls_v_pln_itc_evk_logreg_%s_sfm.jbl" %
        (band))

    scores = cross_val_multiscore(time_decod, X, y, cv=cv)
    np.save(beamformer_mvpa + "source_cls_v_pln_itc_evk_logreg_%s_sfm.npy" %
            (band), scores)
