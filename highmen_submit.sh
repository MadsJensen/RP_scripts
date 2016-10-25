#!/bin/bash
# Generate a single script
cat <<EOF > ${SCRIPT}
#!/bin/bash
#$ -S /bin/bash
export TERM vt100
# Make sure MNI and CFIN are in the path
PATH=$PATH:/usr/local/mni/bin:/usr/local/cfin/bin:.
# Change to current directory.

for sub in $(<subjects.txt)
do
    python $1 $sub 
done

EOF

# Make the new script executable
    chmod u+x ${SCRIPT}

# Finally submit it to the cluster in the long.q queue
    # qsub -j y -q highmem.q -l h_vmem=$2 ${SCRIPT}

