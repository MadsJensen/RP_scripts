"""
Doc string here.

@author mje
@email: mads [] cnru.dk

"""
import sys
import subprocess

cmd = "/usr/local/common/meeg-cfin/configurations/bin/submit_to_isis"

if len(sys.argv) == 4:
    cpu_number = sys.argv[3]
else:
    cpu_number = 4

submit_cmd = "python %s %s" % (sys.argv[1], sys.argv[2])
subprocess.call([cmd, "%s" % cpu_number, submit_cmd])
