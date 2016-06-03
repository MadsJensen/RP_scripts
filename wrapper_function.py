"""
Doc string here.

@author mje
@email: mads [] cnru.dk

"""
import sys
import subprocess

cmd = "/usr/local/common/meeg-cfin/configurations/bin/submit_to_isis"

# subjects = ["0008", "0009", "0010", "0011", "0012", "0013",
#            "0014", "0015","0016", "0017", "0018",  "0019", "0020",
#            "0021", "0022"]

subjects = ["0009", "0010", "0011", "0012", "0013",
            "0014", "0015","0016", "0017", "0018",  "0019", "0020",
            "0021", "0022"]
# TODO: fix subjects


if len(sys.argv) == 3:
    cpu_number = sys.argv[2]
else:
    cpu_number = 4


for subject in subjects[2:]:
    submit_cmd = "python %s %s" % (sys.argv[1], subject)
    subprocess.call([cmd, "%s" % cpu_number, submit_cmd])
