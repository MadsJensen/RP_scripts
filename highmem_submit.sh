


for sub in $(<subjects.txt)
do
    
    OUTPUT=/projects/MINDLAB2011_24-MEG-readiness/scripts/RP_scripts/highmem_logs/log_${sub}.txt
    SCRIPT=/projects/MINDLAB2011_24-MEG-readiness/scripts/RP_scripts/highmem_scripts/script_${sub}.sh
    SCRIPT_PATH=/projects/MINDLAB2011_24-MEG-readiness/scripts/RP_scripts

cat <<EOF > ${SCRIPT}
#!/bin/bash
#$ -S /bin/bash
export TERM vt100
# Make sure MNI and CFIN are in the path
PATH=$PATH:/usr/local/mni/bin:/usr/local/cfin/bin:. 
# Setup freesurfer

python $SCRIPT_PATH/$1 ${sub}

EOF
# Make the new script executable
    chmod u+x ${SCRIPT}
    
    # qsub -q highmem.q -l h_vmem=$2 ${SCRIPT}

done 
