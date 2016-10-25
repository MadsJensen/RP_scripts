
if [ $# == 2 ]
  then
    for sub in $(<subjects.txt)
    do
        eval submit_to_cluster -q $2 \"python $1 $sub\"
    done
    
elif [ $# == 3 ]
  then
    for sub in $(<subjects.txt)
    do
        eval submit_to_cluster -q $2 \"h_vmem=$3 python $1 $sub\"
    done

else
    for sub in $(<subjects.txt)
    do
        eval submit_to_cluster \"python $1 $sub\"
    done

fi
