import sys
import h5io
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression, LassoCV
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel

from mne.decoding import (SlidingEstimator, cross_val_multiscore, LinearModel)

from my_settings import erf_mvpa
windows_size = 10

n_jobs = int(sys.argv[1])

seed = 352341561
seed_cv = 2423423
tol = 1e-5

Xy = h5io.read_hdf5(erf_mvpa + "Xy_cls_v_pln_erf_RM.hd5")
X = Xy['X'][:, :, windows_size:-windows_size]
y = Xy['y']

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
cv_lss = StratifiedKFold(n_splits=4, shuffle=True, random_state=seed_cv)

clf = make_pipeline(
    StandardScaler(),  # z-score normalization
    SelectFromModel(LassoCV(cv=cv_lss, tol=tol, normalize=False)),
    LinearModel(LogisticRegression(C=1, tol=tol, solver='lbfgs')))

time_decod = SlidingEstimator(clf, n_jobs=n_jobs, scoring='roc_auc')

time_decod.fit(X, y)
joblib.dump(time_decod, erf_mvpa + "source_cls_v_pln_evk_logreg_erf_lss.jbl")

scores = cross_val_multiscore(time_decod, X, y, cv=cv)
h5io.write_hdf5(
    erf_mvpa + "source_cls_v_pln_evk_logreg_erf_lss.hd5",
    scores,
    overwrite=True)
