"""
Doc string here.

@author mje
@email: mads [] cnru.dk

"""
import sys
import subprocess


if len(sys.argv) == 4:
    cpu_number = sys.argv[3]
else:
    cpu_number = 1

submit_cmd = 'submit_to_cluster \"python %s %s\"' % (sys.argv[1], cpu_number)
print(submit_cmd)

# subprocess.call([cmd, "%s" % cpu_number, submit_cmd])
