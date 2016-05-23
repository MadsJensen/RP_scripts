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

# path to submit_to_isis
cmd = "/usr/local/common/meeg-cfin/configurations/bin/submit_to_isis"
proj_code = "MINDLAB2011_24-MEG-readiness"

conditions = {"classic": "2", "plan": "4", "interupt": "3"}

db = Query(proj_code)
proj_folder = os.path.join('/projects', proj_code)

script_dir = proj_folder + '/scripts/'

included_subjects = db.get_subjects()
# just test with first one!
included_subjects = included_subjects[6:7]

for sub in included_subjects:
    # this is an example of getting the DICOM files as a list
#    # sequence_name='t1_mprage_3D_sag'
    MEG_study = db.get_studies(sub, modality='MEG')

    for condition in conditions:
        in_name = db.get_files(sub, MEG_study[0], 'MEG',
                                   conditions[condition])
        out_name = "%s_xscan-bads_%s.txt" % (sub[:4], condition)
            
        out_fname = maxfiltered_folder + out_name
        
        xscan_cmd = "xscan -f %s -corr 0.95 > %s" % (in_name[0], out_fname)        
        print(xscan_cmd)

        subprocess.call([cmd, "4", xscan_cmd])
