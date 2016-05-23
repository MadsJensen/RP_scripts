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

ctc = "/projects/MINDLAB2011_24-MEG-readiness/misc/ct_sparse_Mar11-May13.fif"
cal = "/projects/MINDLAB2011_24-MEG-readiness/misc/sss_cal_Mar11-May13.dat"

# MAXFILTER PARAMS #
mf_params = dict(origin='0 0 40',
                 frame='head',
                 autobad="on",
                 st=True,
                 st_buflen=30,
                 st_corr=0.95,
                 movecomp=True,
                 cal=cal,
                 ctc=ctc,
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

conditions = {"classic": "2", "plan": "4", "interupt": "3"}

included_subjects = db.get_subjects()
# just test with first one!
included_subjects = included_subjects[6:]

for sub in included_subjects:
    MEG_study = db.get_studies(sub, modality='MEG')

    for condition in conditions:
        in_name = db.get_files(sub, MEG_study[0], 'MEG',
                               conditions[condition])
        out_name = "%s_%s_mc_tsss-raw.fif" % (sub[:4], condition)
        out_fname = maxfiltered_folder + out_name


            # print(out_fname)
        tsss_mc_log = out_fname[:-3] + "log"
        headpos_log = out_fname[:-4] + "_hp.log"

        # print(tsss_mc_log)
        # print(headpos_log)

        mf_params["logfile"] = tsss_mc_log
#        mf_params["mv_hp"] = headpos_log
        mf = Maxfilter(proj_code)
        mf.build_maxfilter_cmd(in_name[0], out_fname, **mf_params)

        mf.submit_to_isis(n_jobs=4)
        # subprocess.call([cmd, "4", mf.cmd])
