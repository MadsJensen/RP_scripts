import sys

import mne

from my_settings import (beamformer_source)

subject = sys.argv[1]

fwd_pln = mne.read_forward_solution(
    beamformer_source + "%s_planning_cor-fwd.fif" % subject[:4])
fwd_cls = mne.read_forward_solution(
    beamformer_source + "%s_classic_cor-fwd.fif" % subject[:4])

fwd_avg = mne.average_forward_solutions([fwd_pln, fwd_cls])
mne.write_forward_solution(beamformer_source + "%s_avg_cor-fwd.fif" % subject)
