import sys

import joblib
import mne
from alphacsc import BatchCDL

from my_settings import erf_raw, dict_learning, conditions

random_nr = 58260

subject = sys.argv[1]

condtions = list(conditions.keys())  # only need the keys

# AlphaCSC settings
# Define the shape of the dictionary
n_atoms = 15
n_times_atom = int(round(sfreq * 1.0))  # 1000. ms

for condition in conditions:
    epo = mne.read_epochs(
        erf_raw + "{}_{}_ar_grads_erf-epo.fif".format(subject, condition))
    sfreq = epo.info['sfreq']

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
        n_iter=100,
        eps=1e-4,
        # solver for the z-step
        solver_z="lgcd",
        solver_z_kwargs={
            'tol': 1e-2,
            'max_iter': 1000
        },
        # solver for the d-step
        solver_d='alternate_adaptive',
        solver_d_kwargs={'max_iter': 300},
        # Technical parameters
        verbose=1,
        random_state=random_nr,
        n_jobs=1)

    X = epo.get_data()

    cdl.fit(X)

    joblib.dump(dict_learning +
                '{}_{}_ar_grads_csc.jbl'.format(subject, condition))
