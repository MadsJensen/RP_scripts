
if [ $# == 2 ]
  then
    for sub in "0008" "0009" "0010" "0012" "0013" "0014" "0015" "0016" "0017" "0018" "0019" "0020" "0021" "0022"
    do
        submit_to_cluster -q $2 \"python $1 $sub\"
    done

else
    for sub in "0008" "0009" "0010" "0012" "0013" "0014" "0015" "0016" "0017" "0018" "0019" "0020" "0021" "0022"
    do
        submit_to_cluster \"python $1 $sub\"
    done

fi
