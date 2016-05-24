"""
Doc string here.

@author mje
@email: mads [] cnru.dk

"""
import sys
import subprocess

cmd = "/usr/local/common/meeg-cfin/configurations/bin/submit_to_isis"

# subjects_select = ["0005", "0006", "0007", "0008", "0009", "0010",
                   "0011", "0015", "0016", "0017", "0020", "0021",
                   "0022", "0024", "0025"]
# TODO: fix subjects


if len(sys.argv) == 3:
    cpu_number = sys.argv[2]
else:
    cpu_number = 4


for subject in subjects_select:
    submit_cmd = "python %s %s" % (sys.argv[1], subject)
    subprocess.call([cmd, "%s" % cpu_number, submit_cmd])
