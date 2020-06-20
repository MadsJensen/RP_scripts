import sys

import joblib
import mne
from mne.decoding import Scaler
from alphacsc import BatchCDL

from my_settings import erf_raw, dict_learning, conditions

random_nr = 58260

subject = sys.argv[1]

condtions = list(conditions.keys())  # only need the keys

for condition in conditions:
    epo = mne.read_epochs(
        erf_raw +
        "{}_{}_ar_grads_erf_hg-epo.fif".format(subject[:4], condition))
    sfreq = epo.info['sfreq']

    # Define the shape of the dictionary
    n_atoms = 20
    n_times_atom = len(epo.times[::3])

    cdl = BatchCDL(
        # Shape of the dictionary
        n_atoms=n_atoms,
        n_times_atom=n_times_atom,
        # Request a rank1 dictionary with unit norm temporal and spatial maps
        rank1=True,
        uv_constraint='separate',
        # Initialize the dictionary with random chunk from the data
        D_init='chunk',
        # rescale the regularization parameter to be 20% of lambda_max
        lmbd_max="scaled",
        reg=.2,
        # Number of iteration for the alternate minimization and cvg threshold
        n_iter=500,
        eps=1e-4,
        # solver for the z-step
        solver_z="lgcd",
        solver_z_kwargs={
            'tol': 1e-2,
            'max_iter': 5000
        },
        # solver for the d-step
        solver_d='alternate_adaptive',
        solver_d_kwargs={'max_iter': 500},
        sort_atoms=True,
        # Technical parameters
        verbose=1,
        random_state=random_nr,
        n_jobs=1)

    std_scl = Scaler(scalings="mean")
    X = epo.get_data()
    X = std_scl.fit_transform(X)

    cdl.fit(X)

    joblib.dump(
        cdl, dict_learning +
        '{}_{}_ar_grads_hg_std_csc.jbl'.format(subject[:4], condition))
