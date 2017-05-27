import os
from my_settings import mf_autobad_off_folder

from stormdb.access import Query
from stormdb.process import Maxfilter

ctc = "/projects/MINDLAB2011_24-MEG-readiness/misc/ct_sparse_Mar11-May13.fif"
cal = "/projects/MINDLAB2011_24-MEG-readiness/misc/sss_cal_Mar11-May13.dat"

# MAXFILTER PARAMS #
mf_params = dict(
    origin='0 0 40',
    frame='head',
    autobad="off",
    st=True,
    st_buflen=30,
    st_corr=0.95,
    movecomp=True,
    cal=cal,
    ctc=ctc,
    mx_args='',
    maxfilter_bin='maxfilter',
    force=True)

mfopts = dict(
    origin='0 0 40',  # {:.1f} {:.1f} {:.1f}'.format(*tuple(origin_head)),
    frame='head',
    force=True,  # overwrite if needed
    autobad='off',  # or use xscan first
    st=True,  # use tSSS
    st_buflen=30,  # parameter set in beg. of notebook
    st_corr=95,  # parameter set in beg. of notebook
    movecomp=True,
    trans=None,  # compensate to mean initial head position (saved to file),
    # or use None for initial head position
    logfile=None,  # we replace this in each loop
    hp=None,  # head positions, replace in each loop
    n_threads=4  # number of parallel threads to run on
)

# path to submit_to_isis
proj_code = "MINDLAB2011_24-MEG-readiness"

db = Query(proj_code)
proj_folder = os.path.join('/projects', proj_code)

script_dir = proj_folder + '/scripts/RP_scripts'

conditions = {"classic": "2", "plan": "4", "interupt": "3"}

included_subjects = db.get_subjects()
# just test with first one!
included_subjects = included_subjects[7]

for sub in [included_subjects]:
    MEG_study = db.get_studies(sub, modality='MEG')

    for condition in conditions:
        in_name = db.get_files(sub, MEG_study[0], 'MEG',
                               conditions[condition])[0]
        out_name = "%s_%s_mc_tsss-raw.fif" % (sub[:4], condition)
        out_fname = mf_autobad_off_folder + out_name

        # print(out_fname)
        tsss_mc_log = out_fname[:-3] + "log"
        headpos_log = out_fname[:-4] + "_hp.log"

        # print(tsss_mc_log)
        # print(headpos_log)

        mfopts["logfile"] = tsss_mc_log
        #        mf_params["mv_hp"] = headpos_log
        mf = Maxfilter(proj_code)
        mf.build_cmd(in_name, out_fname, **mfopts)

        # subprocess.call([cmd, "4", mf.cmd])
