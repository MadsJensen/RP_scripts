from __future__ import print_function
import os
import sys
import subprocess

from my_settings import *
#
path_to_stormdb = "/usr/local/common/meeg-cfin/stormdb"
sys.path.append(path_to_stormdb)

from stormdb.access import Query
from stormdb.process import Maxfilter


# MAXFILTER PARAMS #
mf_params = dict(origin='0 0 40',
                 frame='head',
                 autobad="off",
                 st=True,
                 st_buflen=30,
                 st_corr=0.95,
                 movecomp=True,
                 # cal=cal,
                 # ctc=ctc,
                 mx_args='',
                 maxfilter_bin='maxfilter',
                 force=True
                 )


# path to submit_to_isis
cmd = "/usr/local/common/meeg-cfin/configurations/bin/submit_to_isis"
proj_code = "MINDLAB2011_24-MEG-readiness"

db = Query(proj_code)
proj_folder = os.path.join('/projects', proj_code)

script_dir = proj_folder + '/scripts/'

included_subjects = db.get_subjects()
# just test with first one!
included_subjects = included_subjects[-3:]

for sub in included_subjects:
    # this is an example of getting the DICOM files as a list
    # sequence_name='t1_mprage_3D_sag'
    MEG_study = db.get_studies(sub, modality='MEG')
    if len(MEG_study) > 0:
        # This is a 2D list with [series_name, series_number]
        series = db.get_series(sub, MEG_study[0], 'MEG')
        series = series["data"]  # only use the "data" series for now.

    # Change this to be more elegant: check whether any item in series
    # matches sequence_name
        in_name = db.get_files(sub, MEG_study[0], 'MEG', series)
        out_name = "%s_%s_mc_raw_tsss.fif" % (sub[:4], "data")
        # print(out_name)

        # if len(in_name) > 1:
        for j, in_file in enumerate(in_name):
            if j == 0:
                out_fname = maxfiltered_folder + out_name
            else:
                out_fname = maxfiltered_folder \
                            + out_name[:-4] + "-%d.fif" % j

            # print(out_fname)
            tsss_mc_log = out_fname[:-3] + "log"
            headpos_log = out_fname[:-4] + "_hp.log"

            # print(tsss_mc_log)
            # print(headpos_log)

            mf_params["logfile"] = tsss_mc_log
            mf_params["mv_hp"] = headpos_log
            mf = Maxfilter(proj_code)
            mf.build_maxfilter_cmd(in_file, out_fname, **mf_params)

            print(mf.cmd)
            subprocess.call([cmd, "4", mf.cmd])
